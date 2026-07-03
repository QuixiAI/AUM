# AUM-Ø fused sequential global block (v6 §5-§9) — Triton/CUDA port of
# kernels/metal/src/aum_silence.metal (the SPEC; read its header for the design).
#
# Same ABI as the Metal kernels: identical stream/weight/save/dout/demit pack layouts, so the
# entire host plumbing in aum_ssm/ops/metal/silence_metal.py (pack builders, unpack_save, the
# _SilenceFused autograd.Function and the backward GEMM assembly) is reused verbatim — only the
# two km.aum_silence_fwd/bwd calls route here on CUDA.
#
# Kernel shape: one program (CTA) per batch row (the recurrence is sequential in t; rows are
# independent), num_warps=8 cooperating on each token's small matvecs. Metal's threadgroup
# buffers become a tiny per-row global scratch (L1-resident) because Triton register tensors
# cannot be dynamically sliced; tl.debug_barrier() stands in for threadgroup_barrier wherever a
# scratch store is later read (the _vstore helper brackets every store with barriers). Weights
# stay in global memory (7.6MB fp32, L2-resident on GA102); all matvec tiles are sized ≤16K
# elements to stay within the register file.
#
# Geometry is HARDCODED to the reference config, like the Metal:
#   d_model D=512, d_sigma DS=128, d_mu DM=32, heads NH=8, head dim DH=64, J=j_max=2.

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------- geometry ----------------
D = tl.constexpr(512)
DS = tl.constexpr(128)
DM = tl.constexpr(32)
NH = tl.constexpr(8)
DH = tl.constexpr(64)
LN_EPS = tl.constexpr(1e-5)

# stream pack per token: c_mu | c_init | c_gate | c_cand | c_pg | c_press | c_phase | g |
#                        cos_t | sin_t | cos_p | sin_p        (silence_flat.py defines the split)
SW_CMU = tl.constexpr(0)
SW_CINIT = tl.constexpr(SW_CMU.value + DM.value)
SW_CGATE = tl.constexpr(SW_CINIT.value + DS.value)
SW_CCAND = tl.constexpr(SW_CGATE.value + DS.value)
SW_CPG = tl.constexpr(SW_CCAND.value + DS.value)
SW_CPRESS = tl.constexpr(SW_CPG.value + DS.value)
SW_CPHASE = tl.constexpr(SW_CPRESS.value + DS.value)
SW_G = tl.constexpr(SW_CPHASE.value + D.value)
SW_COST = tl.constexpr(SW_G.value + D.value)
SW_SINT = tl.constexpr(SW_COST.value + 256)
SW_COSP = tl.constexpr(SW_SINT.value + 256)
SW_SINP = tl.constexpr(SW_COSP.value + 256)
SW = tl.constexpr(SW_SINP.value + 256)                        # 2720

# save pack per token
SV_GHAT = tl.constexpr(0)
SV_MU = tl.constexpr(SV_GHAT.value + D.value)
SV_ETIL = tl.constexpr(SV_MU.value + DM.value)
SV_SIGST = tl.constexpr(SV_ETIL.value + DM.value)             # (J+1) x DS
SV_RPRED = tl.constexpr(SV_SIGST.value + 3 * DS.value)
SV_RST = tl.constexpr(SV_RPRED.value + D.value)               # (J+1) x D
SV_E = tl.constexpr(SV_RST.value + 3 * D.value)
SV_PI = tl.constexpr(SV_E.value + 3)
SV_W = tl.constexpr(SV_PI.value + 1)
SV_SSTAR = tl.constexpr(SV_W.value + 3)
SV_LN5 = tl.constexpr(SV_SSTAR.value + DS.value)              # mean, rstd
SV_LN1 = tl.constexpr(SV_LN5.value + 2)                       # (mean, rstd) x (J+1)
SV = tl.constexpr(SV_LN1.value + 6)                           # 3151

# weight-pack offsets (build_weight_pack order — must match aum_silence.metal W_*)
W_QPRED = tl.constexpr(0)
W_R = tl.constexpr(W_QPRED.value + D.value * DS.value)
W_HYP = tl.constexpr(W_R.value + D.value * D.value)
W_P = tl.constexpr(W_HYP.value + D.value * DS.value)
W_LN5W = tl.constexpr(W_P.value + D.value * D.value)
W_LN5B = tl.constexpr(W_LN5W.value + D.value)
W_MUE = tl.constexpr(W_LN5B.value + D.value)
W_MUS = tl.constexpr(W_MUE.value + DM.value * D.value)
W_ERR = tl.constexpr(W_MUS.value + DM.value * DS.value)
W_IS = tl.constexpr(W_ERR.value + DM.value * D.value)
W_IET = tl.constexpr(W_IS.value + DS.value * DS.value)
W_IMU = tl.constexpr(W_IET.value + DS.value * DM.value)
W_LN1W = tl.constexpr(W_IMU.value + DS.value * DM.value)
W_LN1B = tl.constexpr(W_LN1W.value + DS.value)
W_QSIG = tl.constexpr(W_LN1B.value + DS.value)
W_GS = tl.constexpr(W_QSIG.value + D.value * DS.value)
W_GET = tl.constexpr(W_GS.value + DS.value * DS.value)
W_GMU = tl.constexpr(W_GET.value + DS.value * DM.value)
W_GR = tl.constexpr(W_GMU.value + DS.value * DM.value)
W_NS = tl.constexpr(W_GR.value + DS.value * D.value)
W_NET = tl.constexpr(W_NS.value + DS.value * DS.value)
W_NMU = tl.constexpr(W_NET.value + DS.value * DM.value)
W_NR = tl.constexpr(W_NMU.value + DS.value * DM.value)
W_QG = tl.constexpr(W_NR.value + DS.value * D.value)
W_PRR = tl.constexpr(W_QG.value + DS.value * DS.value)
W_QR = tl.constexpr(W_PRR.value + DS.value * D.value)
W_PCG = tl.constexpr(W_QR.value + DS.value * DS.value)
W_PCR = tl.constexpr(W_PCG.value + DS.value * DM.value)
W_WPI = tl.constexpr(W_PCR.value + DS.value * DM.value)
W_WPDE = tl.constexpr(W_WPI.value + DS.value)
W_WPDS = tl.constexpr(W_WPDE.value + DS.value)
W_H1S = tl.constexpr(W_WPDS.value + DS.value)
W_H1PI = tl.constexpr(W_H1S.value + 64 * DS.value)
W_H1E = tl.constexpr(W_H1PI.value + 64)
W_H2 = tl.constexpr(W_H1E.value + 64)
W_TOTAL = tl.constexpr(W_H2.value + 64)

# forward scratch (per batch row)
SC_SIGMA = tl.constexpr(0)                                    # carried sigma^*_{t-1}
SC_SIG = tl.constexpr(SC_SIGMA.value + DS.value)              # sigma^0..2
SC_MU = tl.constexpr(SC_SIG.value + 3 * DS.value)
SC_ETIL = tl.constexpr(SC_MU.value + DM.value)
SC_EVEC = tl.constexpr(SC_ETIL.value + DM.value)
SC_LN5 = tl.constexpr(SC_EVEC.value + D.value)
SC_QR = tl.constexpr(SC_LN5.value + D.value)
SC_FWD = tl.constexpr(SC_QR.value + D.value)                  # 2112

# incoming-grad (dout) pack — must match silence_metal.py's cat order
DO_SIG = tl.constexpr(0)                                      # (J+1) x DS
DO_GHAT = tl.constexpr(DO_SIG.value + 3 * DS.value)
DO_RST = tl.constexpr(DO_GHAT.value + D.value)                # (J+1) x D
DO_E = tl.constexpr(DO_RST.value + 3 * D.value)
DO_PI = tl.constexpr(DO_E.value + 3)
DO_W = tl.constexpr(DO_PI.value + 1)
DO_SST = tl.constexpr(DO_W.value + 3)
DO = tl.constexpr(DO_SST.value + DS.value)                    # 2567

# emitted d-vector (demit) pack — must match silence_metal.py _DE
DE_PRE = tl.constexpr(0)                                      # dpre (LN5 input grad)  (D)
DE_GHT = tl.constexpr(DE_PRE.value + D.value)                 # dGhat_tot              (D)
DE_ZMU = tl.constexpr(DE_GHT.value + D.value)                 # dz_mu                  (DM)
DE_ETT = tl.constexpr(DE_ZMU.value + DM.value)                # d e_tilde              (DM)
DE_ZI = tl.constexpr(DE_ETT.value + DM.value)                 # dz_init                (DS)
DE_DS0 = tl.constexpr(DE_ZI.value + DS.value)                 # d sigma^j (LN1 dy)     (DS)x3
DE_ZG0 = tl.constexpr(DE_DS0.value + 3 * DS.value)            # dz_gate j=0,1          (DS)x2
DE_ZN0 = tl.constexpr(DE_ZG0.value + 2 * DS.value)            # dz_cand j=0,1          (DS)x2
DE_QP = tl.constexpr(DE_ZN0.value + 2 * DS.value)             # dq_pred                (D)
DE_QS0 = tl.constexpr(DE_QP.value + D.value)                  # dq_sig j=0..2          (D)x3
DE_DG0 = tl.constexpr(DE_QS0.value + 3 * D.value)             # d dG_j                 (DS)x3
DE_DR0 = tl.constexpr(DE_DG0.value + 3 * DS.value)            # d dR_j                 (DS)x3
DE_MUG = tl.constexpr(DE_DR0.value + 3 * DS.value)            # d muG                  (DS)
DE_MUR = tl.constexpr(DE_MUG.value + DS.value)                # d muR                  (DS)
DE_PPI = tl.constexpr(DE_MUR.value + DS.value)                # d pre_pi               (DS)
DE_ZPI = tl.constexpr(DE_PPI.value + DS.value)                # dz_pi                  (1)
DE_H0 = tl.constexpr(DE_ZPI.value + 1)                        # dh j=0,1               (64)x2
DE_ZH = tl.constexpr(DE_H0.value + 2 * 64)                    # dz_h j=0,1             (2)
DE = tl.constexpr(DE_ZH.value + 2)                            # 5443

# backward scratch (per batch row)
SB_SPREV = tl.constexpr(0)                                    # sigma^*_{t-1}
SB_SIG = tl.constexpr(SB_SPREV.value + DS.value)              # saved sigma^0..2
SB_MU = tl.constexpr(SB_SIG.value + 3 * DS.value)
SB_ETIL = tl.constexpr(SB_MU.value + DM.value)
SB_TMPM = tl.constexpr(SB_ETIL.value + DM.value)              # mu (.) detil (mvt src)
SB_EVEC = tl.constexpr(SB_TMPM.value + DM.value)
SB_QR = tl.constexpr(SB_EVEC.value + D.value)                 # qr / dqr
SB_DR = tl.constexpr(SB_QR.value + D.value)                   # dr_j for the per-head ops
SB_DCARRY = tl.constexpr(SB_DR.value + D.value)               # d sigma^* from token t+1
SC_BWD = tl.constexpr(SB_DCARRY.value + DS.value)             # 2272


# ---------------- cooperative helpers ----------------
@triton.jit
def _vload(ptr, N: tl.constexpr):
    return tl.load(ptr + tl.arange(0, N))


@triton.jit
def _vstore(ptr, v, N: tl.constexpr):
    """Scratch store bracketed by CTA barriers (WAR before, RAW after)."""
    tl.debug_barrier()
    tl.store(ptr + tl.arange(0, N), v)
    tl.debug_barrier()


@triton.jit
def _mv(w_ptr, x_ptr, OUT: tl.constexpr, IN: tl.constexpr, BI: tl.constexpr):
    """W (OUT, IN) row-major at w_ptr times x (IN,) at x_ptr -> (OUT,) register tensor."""
    o = tl.arange(0, OUT)
    acc = tl.zeros([OUT], dtype=tl.float32)
    for i0 in range(0, IN, BI):
        i = i0 + tl.arange(0, BI)
        w = tl.load(w_ptr + o[:, None] * IN + i[None, :])
        x = tl.load(x_ptr + i)
        acc += tl.sum(w * x[None, :], axis=1)
    return acc


@triton.jit
def _mvt(w_ptr, s_ptr, OUT: tl.constexpr, IN: tl.constexpr, BO: tl.constexpr):
    """W^T s: W (OUT, IN) row-major, s (OUT,) at s_ptr -> (IN,) register tensor."""
    i = tl.arange(0, IN)
    acc = tl.zeros([IN], dtype=tl.float32)
    for o0 in range(0, OUT, BO):
        o = o0 + tl.arange(0, BO)
        w = tl.load(w_ptr + o[:, None] * IN + i[None, :])
        s = tl.load(s_ptr + o)
        acc += tl.sum(w * s[:, None], axis=0)
    return acc


@triton.jit
def _rot(q, cos_ptr, sin_ptr, SIGN: tl.constexpr):
    """Per-head rotation ladder on a (D,) register vector: pairs (2i, 2i+1) rotated by the
    cos/sin streams. SIGN=-1 rotates by -phi (the backward's unrotate)."""
    q2 = tl.reshape(q, (D // 2, 2))
    q0, q1 = tl.split(q2)
    c = tl.load(cos_ptr + tl.arange(0, D // 2))
    s = tl.load(sin_ptr + tl.arange(0, D // 2)) * SIGN
    return tl.reshape(tl.interleave(q0 * c - q1 * s, q0 * s + q1 * c), (D,))


@triton.jit
def _state_read(S_ptr, qr_ptr, out_ptr):
    """out[h*DH+p] = sum_n S[h,p,n] * qr[h*DH+n]; result written to a global pointer."""
    tl.debug_barrier()
    for h in tl.static_range(8):
        idx = h * DH * DH + tl.arange(0, DH)[:, None] * DH + tl.arange(0, DH)[None, :]
        S_h = tl.load(S_ptr + idx)
        qr_h = tl.load(qr_ptr + h * DH + tl.arange(0, DH))
        tl.store(out_ptr + h * DH + tl.arange(0, DH), tl.sum(S_h * qr_h[None, :], axis=1))
    tl.debug_barrier()


@triton.jit
def _state_read_t(S_ptr, dr_ptr, out_ptr):
    """out[h*DH+n] = sum_p S[h,p,n] * dr[h*DH+p] (the transposed read)."""
    tl.debug_barrier()
    for h in tl.static_range(8):
        idx = h * DH * DH + tl.arange(0, DH)[:, None] * DH + tl.arange(0, DH)[None, :]
        S_h = tl.load(S_ptr + idx)
        dr_h = tl.load(dr_ptr + h * DH + tl.arange(0, DH))
        tl.store(out_ptr + h * DH + tl.arange(0, DH), tl.sum(S_h * dr_h[:, None], axis=0))
    tl.debug_barrier()


@triton.jit
def _layernorm(v, w_ptr, b_ptr, N: tl.constexpr):
    """LayerNorm over a (N,) register vector; returns (y, mean, rstd). Matches the Metal
    kernel's E[x^2]-mean^2 variance and rsqrt."""
    mean = tl.sum(v) / N
    var = tl.sum(v * v) / N - mean * mean
    rstd = libdevice.rsqrt(var + LN_EPS)
    w = tl.load(w_ptr + tl.arange(0, N))
    b = tl.load(b_ptr + tl.arange(0, N))
    return (v - mean) * rstd * w + b, mean, rstd


@triton.jit
def _layernorm_bwd(dy, vin, w_ptr, mean, rstd, N: tl.constexpr):
    """dv = rstd*(dyw - mean(dyw) - xhat*mean(dyw*xhat)) from saved (mean, rstd)."""
    w = tl.load(w_ptr + tl.arange(0, N))
    dyw = dy * w
    xh = (vin - mean) * rstd
    m1 = tl.sum(dyw) / N
    m2 = tl.sum(dyw * xh) / N
    return rstd * (dyw - m1 - xh * m2)


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _read_j(S_ptr, wp, sig_ptr, st, sc, out_ptr):
    """r = S . R(phi_t) W_qsig sigma^j -> out_ptr (device); clobbers the SC_QR scratch."""
    q = _mv(wp + W_QSIG, sig_ptr, D, DS, 32)
    qr = _rot(q, st + SW_COST, st + SW_SINT, 1)
    _vstore(sc + SC_QR, qr, D)
    _state_read(S_ptr, sc + SC_QR, out_ptr)


@triton.jit
def _revise_j(wp, st, sv, sc, J: tl.constexpr):
    """sigma^{J+1} = LN(sigma^J + gate (.) cand); stores sig scratch, save pack, LN stats."""
    zg = (_vload(st + SW_CGATE, DS)
          + _mv(wp + W_GS, sc + SC_SIG + J * DS, DS, DS, 128)
          + _mv(wp + W_GET, sc + SC_ETIL, DS, DM, 32)
          + _mv(wp + W_GMU, sc + SC_MU, DS, DM, 32)
          + _mv(wp + W_GR, sv + SV_RST + J * D, DS, D, 128))
    zn = (_vload(st + SW_CCAND, DS)
          + _mv(wp + W_NS, sc + SC_SIG + J * DS, DS, DS, 128)
          + _mv(wp + W_NET, sc + SC_ETIL, DS, DM, 32)
          + _mv(wp + W_NMU, sc + SC_MU, DS, DM, 32)
          + _mv(wp + W_NR, sv + SV_RST + J * D, DS, D, 128))
    gate = _sigmoid(zg)
    cand = libdevice.tanh(zn)
    v = _vload(sc + SC_SIG + J * DS, DS) + gate * cand
    y, m, r = _layernorm(v, wp + W_LN1W, wp + W_LN1B, DS)
    tl.store(sv + SV_LN1 + 2 * (J + 1), m)
    tl.store(sv + SV_LN1 + 2 * (J + 1) + 1, r)
    tl.store(sv + SV_SIGST + (J + 1) * DS + tl.arange(0, DS), y)
    _vstore(sc + SC_SIG + (J + 1) * DS, y, DS)


@triton.jit
def _E_j(wp, st, sv, sc, muG, muR, sig_prev, kappa, J: tl.constexpr):
    """E_j = |muG(.)dG|^2 + |muR(.)dR|^2 + kappa|sigma^j - sigma_prev|^2; returns (E_j, dR_j)."""
    sj = _vload(sc + SC_SIG + J * DS, DS)
    dG = _vload(st + SW_CPG, DS) - _mv(wp + W_QG, sc + SC_SIG + J * DS, DS, DS, 128)
    dR = (_mv(wp + W_PRR, sv + SV_RST + J * D, DS, D, 128)
          - _mv(wp + W_QR, sc + SC_SIG + J * DS, DS, DS, 128))
    dsv = sj - sig_prev
    Ej = tl.sum(muG * muG * dG * dG + muR * muR * dR * dR + kappa * dsv * dsv)
    tl.store(sv + SV_E + J, Ej)
    return Ej, dR


@triton.jit
def _halt_h(wp, sig_ptr, pi, Ej):
    """h_j = tanh(Wh1_s sigma^j + wh1_pi*pi + wh1_E*E_j) — the halt MLP hidden vector."""
    hv = (_mv(wp + W_H1S, sig_ptr, 64, DS, 128)
          + _vload(wp + W_H1PI, 64) * pi + _vload(wp + W_H1E, 64) * Ej)
    return libdevice.tanh(hv)


@triton.jit
def _halt_p(wp, sig_ptr, pi, Ej):
    """p_j = sigmoid(w_h2 . h_j)."""
    h = _halt_h(wp, sig_ptr, pi, Ej)
    return _sigmoid(tl.sum(_vload(wp + W_H2, 64) * h))


# ---------------- forward ----------------
@triton.jit
def _silence_fwd_kernel(
        streams, alpha, xw, k_rot, halt_u, wp, S, save, j_star, S_ckpt, scratch,
        L, kappa, forced, delta,
        FORCED: tl.constexpr, NO_OP: tl.constexpr, HALT2: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    Sb = S + b * (NH * DH * DH)
    sc = scratch + b * SC_FWD
    nseg = (L + 63) // 64

    _vstore(sc + SC_SIGMA, tl.zeros([DS], dtype=tl.float32), DS)

    for t in range(0, L):
        st = streams + (b * L + t) * SW
        sv = save + (b * L + t) * SV
        xt = xw + (b * L + t) * (NH * DH)
        kt = k_rot + (b * L + t) * (NH * DH)
        at = alpha + (b * L + t) * NH

        if t % 64 == 0:                          # checkpoint S_{t-1} for the backward replay
            ck = S_ckpt + (b * nseg + t // 64) * (NH * DH * DH)
            for h in tl.static_range(8):
                idx = h * DH * DH + tl.arange(0, DH)[:, None] * DH + tl.arange(0, DH)[None, :]
                tl.store(ck + idx, tl.load(Sb + idx))
            tl.debug_barrier()

        # ---- predictive read against S_{t-1} (BEFORE the state update) ----
        q = _mv(wp + W_QPRED, sc + SC_SIGMA, D, DS, 32)
        qr = _rot(q, st + SW_COSP, st + SW_SINP, 1)
        _vstore(sc + SC_QR, qr, D)
        _state_read(Sb, sc + SC_QR, sv + SV_RPRED)

        # ---- state update S = alpha (.) S + x (x) k ----
        for h in tl.static_range(8):
            a_h = tl.load(at + h)
            x_h = tl.load(xt + h * DH + tl.arange(0, DH))
            k_h = tl.load(kt + h * DH + tl.arange(0, DH))
            idx = h * DH * DH + tl.arange(0, DH)[:, None] * DH + tl.arange(0, DH)[None, :]
            tl.store(Sb + idx, a_h * tl.load(Sb + idx) + x_h[:, None] * k_h[None, :])
        tl.debug_barrier()

        # ---- g_hat = W_P LN512(W_R r_pred + W_hyp sigma + c_phase) ; e = g - g_hat ----
        pre = (_vload(st + SW_CPHASE, D)
               + _mv(wp + W_R, sv + SV_RPRED, D, D, 32)
               + _mv(wp + W_HYP, sc + SC_SIGMA, D, DS, 32))
        y5, m5, r5 = _layernorm(pre, wp + W_LN5W, wp + W_LN5B, D)
        tl.store(sv + SV_LN5, m5)
        tl.store(sv + SV_LN5 + 1, r5)
        _vstore(sc + SC_LN5, y5, D)
        ghat = _mv(wp + W_P, sc + SC_LN5, D, D, 32)
        tl.store(sv + SV_GHAT + tl.arange(0, D), ghat)
        evec = _vload(st + SW_G, D) - ghat
        _vstore(sc + SC_EVEC, evec, D)

        # ---- mu, e_tilde ----
        zmu = (_vload(st + SW_CMU, DM)
               + _mv(wp + W_MUE, sc + SC_EVEC, DM, D, 512)
               + _mv(wp + W_MUS, sc + SC_SIGMA, DM, DS, 128))
        mu = _sigmoid(zmu)
        etil = mu * _mv(wp + W_ERR, sc + SC_EVEC, DM, D, 512)
        tl.store(sv + SV_MU + tl.arange(0, DM), mu)
        tl.store(sv + SV_ETIL + tl.arange(0, DM), etil)
        _vstore(sc + SC_MU, mu, DM)
        _vstore(sc + SC_ETIL, etil, DM)

        # ---- sigma^0 ----
        z0 = (_vload(st + SW_CINIT, DS)
              + _mv(wp + W_IS, sc + SC_SIGMA, DS, DS, 128)
              + _mv(wp + W_IET, sc + SC_ETIL, DS, DM, 32)
              + _mv(wp + W_IMU, sc + SC_MU, DS, DM, 32))
        s0, m1, r1 = _layernorm(z0, wp + W_LN1W, wp + W_LN1B, DS)
        tl.store(sv + SV_LN1, m1)
        tl.store(sv + SV_LN1 + 1, r1)
        tl.store(sv + SV_SIGST + tl.arange(0, DS), s0)
        _vstore(sc + SC_SIG, s0, DS)

        # ---- revision loop: r_0 -> revise(0) -> r_1 -> revise(1) -> r_2 ----
        _read_j(Sb, wp, sc + SC_SIG, st, sc, sv + SV_RST)
        _revise_j(wp, st, sv, sc, 0)
        _read_j(Sb, wp, sc + SC_SIG + DS, st, sc, sv + SV_RST + D)
        _revise_j(wp, st, sv, sc, 1)
        _read_j(Sb, wp, sc + SC_SIG + 2 * DS, st, sc, sv + SV_RST + 2 * D)

        # ---- precision-weighted consistency E_j; pressure pi ----
        muG = _mv(wp + W_PCG, sc + SC_MU, DS, DM, 32)
        muR = _mv(wp + W_PCR, sc + SC_MU, DS, DM, 32)
        sig_prev = _vload(sc + SC_SIGMA, DS)
        E0, dR0 = _E_j(wp, st, sv, sc, muG, muR, sig_prev, kappa, 0)
        E1, _ = _E_j(wp, st, sv, sc, muG, muR, sig_prev, kappa, 1)
        E2, _ = _E_j(wp, st, sv, sc, muG, muR, sig_prev, kappa, 2)
        dsR = tl.sqrt(tl.sum(dR0 * dR0))
        de_n = tl.sqrt(tl.sum(etil * etil))
        pre_pi = (_vload(st + SW_CPRESS, DS) + _vload(wp + W_WPDE, DS) * de_n
                  + _vload(wp + W_WPDS, DS) * dsR)
        z = tl.sum(_vload(wp + W_WPI, DS) * libdevice.tanh(pre_pi))
        pi = tl.where(z > 20.0, z, tl.log(1.0 + tl.exp(z)))
        tl.store(sv + SV_PI, pi)

        # ---- halting: p_j -> w_j -> j* ; carry sigma <- sigma^{j*} ----
        p0 = _halt_p(wp, sc + SC_SIG, pi, E0)
        p1 = _halt_p(wp, sc + SC_SIG + DS, pi, E1)
        w0 = p0
        w1 = p1 * (1.0 - p0)
        w2 = (1.0 - p0) * (1.0 - p1)
        if FORCED:
            js = tl.zeros((), dtype=tl.int32) + forced   # forced may be constexpr-specialized
            w0 = tl.where(js == 0, 1.0, 0.0)
            w1 = tl.where(js == 1, 1.0, 0.0)
            w2 = tl.where(js == 2, 1.0, 0.0)
        elif HALT2:                              # inference: first p_j >= delta (p_J = 1)
            js = tl.where(p0 >= delta, 0, tl.where(p1 >= delta, 1, 2)).to(tl.int32)
        else:
            u = tl.load(halt_u + b * L + t)
            js = tl.where(w0 >= u, 0, tl.where(w0 + w1 >= u, 1, 2)).to(tl.int32)
        tl.store(sv + SV_W + 0, w0)
        tl.store(sv + SV_W + 1, w1)
        tl.store(sv + SV_W + 2, w2)
        tl.store(j_star + b * L + t, js)
        if NO_OP:                                # revision discarded, sigma^0 carried
            js = js * 0
        sig_star = tl.load(sc + SC_SIG + js * DS + tl.arange(0, DS))
        tl.store(sv + SV_SSTAR + tl.arange(0, DS), sig_star)
        _vstore(sc + SC_SIGMA, sig_star, DS)


def aum_silence_fwd(streams, alpha, xw, k_rot, halt_u, wpack, kappa, forced=-1, no_op=False,
                    halt_mode=0, delta=0.5):
    """Same contract as kernels.metal.aum_silence_fwd: returns (save (B,L,3151),
    j_star (B,L) int32, S_final (B,8,64,64), S_ckpt (B,ceil(L/64),8,64,64)), all fp32."""
    B, L = streams.shape[:2]
    dev = streams.device
    f32 = lambda x: x.contiguous().float()  # noqa: E731
    streams, alpha, xw, k_rot = f32(streams), f32(alpha), f32(xw), f32(k_rot)
    halt_u, wpack = f32(halt_u), f32(wpack)
    assert streams.shape[-1] == SW.value and wpack.numel() == W_TOTAL.value
    nseg = (L + 63) // 64
    save = torch.empty(B, L, SV.value, device=dev, dtype=torch.float32)
    j_star = torch.empty(B, L, device=dev, dtype=torch.int32)
    S = torch.zeros(B, NH.value, DH.value, DH.value, device=dev, dtype=torch.float32)
    S_ckpt = torch.empty(B, nseg, NH.value, DH.value, DH.value, device=dev,
                         dtype=torch.float32)
    scratch = torch.empty(B, SC_FWD.value, device=dev, dtype=torch.float32)
    _silence_fwd_kernel[(B,)](
        streams, alpha, xw.reshape(B, L, -1), k_rot.reshape(B, L, -1), halt_u, wpack,
        S, save, j_star, S_ckpt, scratch, L, float(kappa), int(forced), float(delta),
        FORCED=int(forced) >= 0, NO_OP=bool(no_op), HALT2=int(halt_mode) == 2,
        num_warps=8)
    return save, j_star, S, S_ckpt


# ---------------------------------------------------------------------------
# BACKWARD — reverse march with exact per-token replay (see aum_silence.metal's
# backward header for the full design: segment S-replay off the forward checkpoints, demit
# emission instead of in-kernel weight grads, dalpha/dxw/dkrot in-kernel).

@triton.jit
def _mv_reg(w_ptr, x, OUT: tl.constexpr, IN: tl.constexpr):
    """W (OUT, IN) times a small REGISTER vector x (IN,) — full tile, so OUT*IN must be small."""
    o = tl.arange(0, OUT)
    i = tl.arange(0, IN)
    w = tl.load(w_ptr + o[:, None] * IN + i[None, :])
    return tl.sum(w * x[None, :], axis=1)


@triton.jit
def _dR_j(wp, sv, sc, J: tl.constexpr):
    """dR_j = P_R r_j - Q_R sigma^j (recomputed from the save pack / sig scratch)."""
    return (_mv(wp + W_PRR, sv + SV_RST + J * D, DS, D, 128)
            - _mv(wp + W_QR, sc + SB_SIG + J * DS, DS, DS, 128))


@triton.jit
def _halt_bwd_j(wp, sc, de, pi_v, Ej, dz, J: tl.constexpr):
    """Emit dh_j; return (dpi_contrib, dE_contrib, dsig_contrib)."""
    hv = _halt_h(wp, sc + SB_SIG + J * DS, pi_v, Ej)
    dh = (1.0 - hv * hv) * _vload(wp + W_H2, 64) * dz
    _vstore(de + DE_H0 + J * 64, dh, 64)
    dpi_c = tl.sum(_vload(wp + W_H1PI, 64) * dh)
    dE_c = tl.sum(_vload(wp + W_H1E, 64) * dh)
    dsig_c = _mvt(wp + W_H1S, de + DE_H0 + J * 64, 64, DS, 64)
    return dpi_c, dE_c, dsig_c


@triton.jit
def _consistency_bwd_j(wp, st, sv, sc, de, muG, muR, dR, dEj, sprev, kappa, fold,
                       dsj, dsprev, dmuG, dmuR, drj, J: tl.constexpr):
    """One j of the consistency backward (mu detached). fold = the dsR scale (j=0 only).
    Returns updated (dsj, dsprev, dmuG, dmuR, drj)."""
    sj = _vload(sc + SB_SIG + J * DS, DS)
    dG = _vload(st + SW_CPG, DS) - _mv(wp + W_QG, sc + SB_SIG + J * DS, DS, DS, 128)
    ddG = 2.0 * muG * muG * dG * dEj
    ddR = 2.0 * muR * muR * dR * dEj + fold * dR
    dmuG += 2.0 * muG * dG * dG * dEj
    dmuR += 2.0 * muR * dR * dR * dEj
    _vstore(de + DE_DG0 + J * DS, ddG, DS)
    _vstore(de + DE_DR0 + J * DS, ddR, DS)
    dk = 2.0 * kappa * (sj - sprev) * dEj
    dsj += dk
    dsprev -= dk
    dsj -= (_mvt(wp + W_QG, de + DE_DG0 + J * DS, DS, DS, 128)
            + _mvt(wp + W_QR, de + DE_DR0 + J * DS, DS, DS, 128))
    drj += _mvt(wp + W_PRR, de + DE_DR0 + J * DS, DS, D, 32)
    return dsj, dsprev, dmuG, dmuR, drj


@triton.jit
def _read_bwd_j(S_ptr, dS_ptr, wp, cos_ptr, sin_ptr, wq_off: tl.constexpr, sig_ptr, sc,
                dq_out_ptr, drj, dsj):
    """Backward of r = S . R(phi) W_q sig: dS += dr (x) qr; emits dq; returns dsj + W_q^T dq.
    Clobbers SB_QR / SB_DR."""
    q = _mv(wp + wq_off, sig_ptr, D, DS, 32)
    qr = _rot(q, cos_ptr, sin_ptr, 1)
    _vstore(sc + SB_QR, qr, D)
    _vstore(sc + SB_DR, drj, D)
    for h in tl.static_range(8):
        idx = h * DH * DH + tl.arange(0, DH)[:, None] * DH + tl.arange(0, DH)[None, :]
        drh = tl.load(sc + SB_DR + h * DH + tl.arange(0, DH))
        qrh = tl.load(sc + SB_QR + h * DH + tl.arange(0, DH))
        tl.store(dS_ptr + idx, tl.load(dS_ptr + idx) + drh[:, None] * qrh[None, :])
    tl.debug_barrier()
    _state_read_t(S_ptr, sc + SB_DR, sc + SB_QR)              # dqr -> SB_QR
    dq = _rot(_vload(sc + SB_QR, D), cos_ptr, sin_ptr, -1)
    _vstore(dq_out_ptr, dq, D)
    return dsj + _mvt(wp + wq_off, dq_out_ptr, D, DS, 128)


@triton.jit
def _revise_bwd_j(wp, st, sv, sc, de, ds_next, ds_cur, detil, dmu, dr_cur, JR: tl.constexpr):
    """Backward of sigma^{JR+1} = LN(sigma^JR + gate (.) cand), consuming the complete
    ds_next. Returns updated (ds_cur, detil, dmu, dr_cur)."""
    zg = (_vload(st + SW_CGATE, DS)
          + _mv(wp + W_GS, sc + SB_SIG + JR * DS, DS, DS, 128)
          + _mv(wp + W_GET, sc + SB_ETIL, DS, DM, 32)
          + _mv(wp + W_GMU, sc + SB_MU, DS, DM, 32)
          + _mv(wp + W_GR, sv + SV_RST + JR * D, DS, D, 128))
    zn = (_vload(st + SW_CCAND, DS)
          + _mv(wp + W_NS, sc + SB_SIG + JR * DS, DS, DS, 128)
          + _mv(wp + W_NET, sc + SB_ETIL, DS, DM, 32)
          + _mv(wp + W_NMU, sc + SB_MU, DS, DM, 32)
          + _mv(wp + W_NR, sv + SV_RST + JR * D, DS, D, 128))
    gate = _sigmoid(zg)
    cand = libdevice.tanh(zn)
    v = _vload(sc + SB_SIG + JR * DS, DS) + gate * cand       # pre-LN input
    _vstore(de + DE_DS0 + (JR + 1) * DS, ds_next, DS)         # emit the LN output grad
    mean = tl.load(sv + SV_LN1 + 2 * (JR + 1))
    rstd = tl.load(sv + SV_LN1 + 2 * (JR + 1) + 1)
    dvv = _layernorm_bwd(ds_next, v, wp + W_LN1W, mean, rstd, DS)
    ds_cur += dvv                                             # through the residual
    dzg = gate * (1.0 - gate) * (dvv * cand)
    dzn = (1.0 - cand * cand) * (dvv * gate)
    _vstore(de + DE_ZG0 + JR * DS, dzg, DS)
    _vstore(de + DE_ZN0 + JR * DS, dzn, DS)
    ds_cur += (_mvt(wp + W_GS, de + DE_ZG0 + JR * DS, DS, DS, 128)
               + _mvt(wp + W_NS, de + DE_ZN0 + JR * DS, DS, DS, 128))
    detil += (_mvt(wp + W_GET, de + DE_ZG0 + JR * DS, DS, DM, 128)
              + _mvt(wp + W_NET, de + DE_ZN0 + JR * DS, DS, DM, 128))
    dmu += (_mvt(wp + W_GMU, de + DE_ZG0 + JR * DS, DS, DM, 128)
            + _mvt(wp + W_NMU, de + DE_ZN0 + JR * DS, DS, DM, 128))
    dr_cur += (_mvt(wp + W_GR, de + DE_ZG0 + JR * DS, DS, D, 32)
               + _mvt(wp + W_NR, de + DE_ZN0 + JR * DS, DS, D, 32))
    return ds_cur, detil, dmu, dr_cur


@triton.jit
def _silence_bwd_kernel(
        streams, alpha, xw, k_rot, wp, save, j_star, S_ckpt, dout, demit,
        dalpha, dxw, dkrot, S_seg, dS, scratch, L, kappa,
        FORCED: tl.constexpr, NO_OP: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    nseg = (L + 63) // 64
    dSb = dS + b * (NH * DH * DH)
    seg = S_seg + b * (65 * NH * DH * DH)         # slot 0 = segment-start ckpt; slot i+1 = S_{t0+i}
    sc = scratch + b * SC_BWD

    _vstore(sc + SB_DCARRY, tl.zeros([DS], dtype=tl.float32), DS)

    for si in range(0, nseg):
        s = nseg - 1 - si
        t0 = s * 64
        tend = tl.minimum(L, t0 + 64)

        # ---- replay S forward through this segment from the checkpoint ----
        ck = S_ckpt + (b * nseg + s) * (NH * DH * DH)
        for h in tl.static_range(8):
            idx = h * DH * DH + tl.arange(0, DH)[:, None] * DH + tl.arange(0, DH)[None, :]
            tl.store(seg + idx, tl.load(ck + idx))
        tl.debug_barrier()
        for i in range(t0, tend):
            prev = seg + (i - t0) * (NH * DH * DH)
            cur = seg + (i - t0 + 1) * (NH * DH * DH)
            xt = xw + (b * L + i) * (NH * DH)
            kt = k_rot + (b * L + i) * (NH * DH)
            at = alpha + (b * L + i) * NH
            for h in tl.static_range(8):
                a_h = tl.load(at + h)
                x_h = tl.load(xt + h * DH + tl.arange(0, DH))
                k_h = tl.load(kt + h * DH + tl.arange(0, DH))
                idx = h * DH * DH + tl.arange(0, DH)[:, None] * DH + tl.arange(0, DH)[None, :]
                tl.store(cur + idx, a_h * tl.load(prev + idx) + x_h[:, None] * k_h[None, :])
            tl.debug_barrier()

        # ---- reverse token march within the segment ----
        for ti in range(0, tend - t0):
            t = tend - 1 - ti
            st = streams + (b * L + t) * SW
            sv = save + (b * L + t) * SV
            dv = dout + (b * L + t) * DO
            de = demit + (b * L + t) * DE
            xt = xw + (b * L + t) * (NH * DH)
            kt = k_rot + (b * L + t) * (NH * DH)
            at = alpha + (b * L + t) * NH
            S_t = seg + (t - t0 + 1) * (NH * DH * DH)
            S_p = seg + (t - t0) * (NH * DH * DH)
            js = tl.load(j_star + b * L + t)
            if NO_OP:
                js = js * 0

            # load saved sigma trajectory / mu / e_tilde / sigma_prev into scratch + registers
            _vstore(sc + SB_SIG, _vload(sv + SV_SIGST, DS), DS)
            _vstore(sc + SB_SIG + DS, _vload(sv + SV_SIGST + DS, DS), DS)
            _vstore(sc + SB_SIG + 2 * DS, _vload(sv + SV_SIGST + 2 * DS, DS), DS)
            tmask = (tl.zeros([DS], dtype=tl.int32) + t) > 0
            sprev = tl.load(save + (b * L + t - 1) * SV + SV_SSTAR + tl.arange(0, DS),
                            mask=tmask, other=0.0)
            _vstore(sc + SB_SPREV, sprev, DS)
            mu = _vload(sv + SV_MU, DM)
            etil = _vload(sv + SV_ETIL, DM)
            _vstore(sc + SB_MU, mu, DM)
            _vstore(sc + SB_ETIL, etil, DM)

            ds0 = _vload(dv + DO_SIG, DS)
            ds1 = _vload(dv + DO_SIG + DS, DS)
            ds2 = _vload(dv + DO_SIG + 2 * DS, DS)
            carry = _vload(sc + SB_DCARRY, DS) + _vload(dv + DO_SST, DS)
            ds0 += tl.where(js == 0, carry, 0.0)
            ds1 += tl.where(js == 1, carry, 0.0)
            ds2 += tl.where(js == 2, carry, 0.0)
            dr0 = _vload(dv + DO_RST, D)
            dr1 = _vload(dv + DO_RST + D, D)
            dr2 = _vload(dv + DO_RST + 2 * D, D)

            pi_v = tl.load(sv + SV_PI)
            dpi = tl.load(dv + DO_PI)
            dE0 = tl.load(dv + DO_E + 0)
            dE1 = tl.load(dv + DO_E + 1)
            dE2 = tl.load(dv + DO_E + 2)

            # ---- halting backward (skipped when forced: w was one-hot) ----
            if FORCED:
                tl.store(de + DE_ZH, 0.0)
                tl.store(de + DE_ZH + 1, 0.0)
                _vstore(de + DE_H0, tl.zeros([128], dtype=tl.float32), 128)
            else:
                E0v = tl.load(sv + SV_E + 0)
                E1v = tl.load(sv + SV_E + 1)
                p0 = _halt_p(wp, sc + SB_SIG, pi_v, E0v)
                p1 = _halt_p(wp, sc + SB_SIG + DS, pi_v, E1v)
                dw0 = tl.load(dv + DO_W + 0)
                dw1 = tl.load(dv + DO_W + 1)
                dw2 = tl.load(dv + DO_W + 2)
                dp0 = dw0 - dw1 * p1 - dw2 * (1.0 - p1)
                dp1 = dw1 * (1.0 - p0) - dw2 * (1.0 - p0)
                dz0 = p0 * (1.0 - p0) * dp0
                dz1 = p1 * (1.0 - p1) * dp1
                tl.store(de + DE_ZH, dz0)
                tl.store(de + DE_ZH + 1, dz1)
                dpi_c, dE_c, ds_c = _halt_bwd_j(wp, sc, de, pi_v, E0v, dz0, 0)
                dpi += dpi_c
                dE0 += dE_c
                ds0 += ds_c
                dpi_c, dE_c, ds_c = _halt_bwd_j(wp, sc, de, pi_v, E1v, dz1, 1)
                dpi += dpi_c
                dE1 += dE_c
                ds1 += ds_c

            # ---- pressure backward ----
            de_n = tl.sqrt(tl.sum(etil * etil))
            dR0 = _dR_j(wp, sv, sc, 0)
            dsR = tl.sqrt(tl.sum(dR0 * dR0))
            dz_pi = (1.0 - tl.exp(-pi_v)) * dpi               # softplus' = sigmoid(z)
            tl.store(de + DE_ZPI, dz_pi)
            pre_pi = (_vload(st + SW_CPRESS, DS) + _vload(wp + W_WPDE, DS) * de_n
                      + _vload(wp + W_WPDS, DS) * dsR)
            th = libdevice.tanh(pre_pi)
            dppi = _vload(wp + W_WPI, DS) * (1.0 - th * th) * dz_pi
            _vstore(de + DE_PPI, dppi, DS)
            d_de = tl.sum(_vload(wp + W_WPDE, DS) * dppi)
            d_ds = tl.sum(_vload(wp + W_WPDS, DS) * dppi)
            detil = tl.where(de_n > 0.0, etil / de_n * d_de, 0.0)
            dmu = tl.zeros([DM], dtype=tl.float32)
            fold = tl.where(dsR > 0.0, d_ds / dsR, 0.0)       # the dsR fold into ddR_0

            # ---- consistency backward (mu detached) ----
            muG = _mv(wp + W_PCG, sc + SB_MU, DS, DM, 32)
            muR = _mv(wp + W_PCR, sc + SB_MU, DS, DM, 32)
            dmuG = tl.zeros([DS], dtype=tl.float32)
            dmuR = tl.zeros([DS], dtype=tl.float32)
            dsprev = tl.zeros([DS], dtype=tl.float32)
            ds0, dsprev, dmuG, dmuR, dr0 = _consistency_bwd_j(
                wp, st, sv, sc, de, muG, muR, dR0, dE0, sprev, kappa, fold,
                ds0, dsprev, dmuG, dmuR, dr0, 0)
            ds1, dsprev, dmuG, dmuR, dr1 = _consistency_bwd_j(
                wp, st, sv, sc, de, muG, muR, _dR_j(wp, sv, sc, 1), dE1, sprev, kappa, 0.0,
                ds1, dsprev, dmuG, dmuR, dr1, 1)
            ds2, dsprev, dmuG, dmuR, dr2 = _consistency_bwd_j(
                wp, st, sv, sc, de, muG, muR, _dR_j(wp, sv, sc, 2), dE2, sprev, kappa, 0.0,
                ds2, dsprev, dmuG, dmuR, dr2, 2)
            _vstore(de + DE_MUG, dmuG, DS)
            _vstore(de + DE_MUR, dmuR, DS)

            # ---- interleaved reverse through the revision loop ----
            # read(2) -> revise(1) -> read(1) -> revise(0) -> read(0) -> init
            ds2 = _read_bwd_j(S_t, dSb, wp, st + SW_COST, st + SW_SINT, W_QSIG,
                              sc + SB_SIG + 2 * DS, sc, de + DE_QS0 + 2 * D, dr2, ds2)
            ds1, detil, dmu, dr1 = _revise_bwd_j(wp, st, sv, sc, de, ds2, ds1, detil, dmu,
                                                 dr1, 1)
            ds1 = _read_bwd_j(S_t, dSb, wp, st + SW_COST, st + SW_SINT, W_QSIG,
                              sc + SB_SIG + DS, sc, de + DE_QS0 + D, dr1, ds1)
            ds0, detil, dmu, dr0 = _revise_bwd_j(wp, st, sv, sc, de, ds1, ds0, detil, dmu,
                                                 dr0, 0)
            ds0 = _read_bwd_j(S_t, dSb, wp, st + SW_COST, st + SW_SINT, W_QSIG,
                              sc + SB_SIG, sc, de + DE_QS0, dr0, ds0)

            # ---- sigma^0 (init) backward ----
            _vstore(de + DE_DS0, ds0, DS)
            z_init = (_vload(st + SW_CINIT, DS)
                      + _mv(wp + W_IS, sc + SB_SPREV, DS, DS, 128)
                      + _mv(wp + W_IET, sc + SB_ETIL, DS, DM, 32)
                      + _mv(wp + W_IMU, sc + SB_MU, DS, DM, 32))
            dzi = _layernorm_bwd(ds0, z_init, wp + W_LN1W, tl.load(sv + SV_LN1),
                                 tl.load(sv + SV_LN1 + 1), DS)
            _vstore(de + DE_ZI, dzi, DS)
            dsprev += _mvt(wp + W_IS, de + DE_ZI, DS, DS, 128)
            detil += _mvt(wp + W_IET, de + DE_ZI, DS, DM, 128)
            dmu += _mvt(wp + W_IMU, de + DE_ZI, DS, DM, 128)

            # ---- e_tilde / mu backward ----
            evec = _vload(st + SW_G, D) - _vload(sv + SV_GHAT, D)
            _vstore(sc + SB_EVEC, evec, D)
            werr_e = _mv(wp + W_ERR, sc + SB_EVEC, DM, D, 512)
            dmu += werr_e * detil
            _vstore(de + DE_ETT, detil, DM)
            _vstore(sc + SB_TMPM, mu * detil, DM)             # d(W_err e)
            dzmu = mu * (1.0 - mu) * dmu
            _vstore(de + DE_ZMU, dzmu, DM)
            devec = (_mvt(wp + W_ERR, sc + SB_TMPM, DM, D, 32)
                     + _mvt(wp + W_MUE, de + DE_ZMU, DM, D, 32))
            dsprev += _mvt(wp + W_MUS, de + DE_ZMU, DM, DS, 32)

            # ---- g_hat backward ----
            dght = _vload(dv + DO_GHAT, D) - devec            # dGhat_tot
            _vstore(de + DE_GHT, dght, D)
            dy5 = _mvt(wp + W_P, de + DE_GHT, D, D, 32)
            pre5 = (_vload(st + SW_CPHASE, D)
                    + _mv(wp + W_R, sv + SV_RPRED, D, D, 32)
                    + _mv(wp + W_HYP, sc + SB_SPREV, D, DS, 32))
            dpre = _layernorm_bwd(dy5, pre5, wp + W_LN5W, tl.load(sv + SV_LN5),
                                  tl.load(sv + SV_LN5 + 1), D)
            _vstore(de + DE_PRE, dpre, D)
            dsprev += _mvt(wp + W_HYP, de + DE_PRE, D, DS, 32)
            dr_pred = _mvt(wp + W_R, de + DE_PRE, D, D, 32)

            # ---- S-chain step at token t: dalpha/dxw/dkrot from dS_t and S_{t-1} ----
            for h in tl.static_range(8):
                idx = h * DH * DH + tl.arange(0, DH)[:, None] * DH + tl.arange(0, DH)[None, :]
                dS_h = tl.load(dSb + idx)
                Sp_h = tl.load(S_p + idx)
                a_h = tl.load(at + h)
                x_h = tl.load(xt + h * DH + tl.arange(0, DH))
                k_h = tl.load(kt + h * DH + tl.arange(0, DH))
                tl.store(dalpha + (b * L + t) * NH + h, tl.sum(dS_h * Sp_h))
                tl.store(dxw + (b * L + t) * (NH * DH) + h * DH + tl.arange(0, DH),
                         tl.sum(dS_h * k_h[None, :], axis=1))
                tl.store(dkrot + (b * L + t) * (NH * DH) + h * DH + tl.arange(0, DH),
                         tl.sum(dS_h * x_h[:, None], axis=0))
                tl.store(dSb + idx, dS_h * a_h)               # dS <- alpha_t (.) dS
            tl.debug_barrier()

            # ---- predictive-read backward (reads S_{t-1}; contributes to dS_{t-1}) ----
            ds_dummy = tl.zeros([DS], dtype=tl.float32)
            dsprev += _read_bwd_j(S_p, dSb, wp, st + SW_COSP, st + SW_SINP, W_QPRED,
                                  sc + SB_SPREV, sc, de + DE_QP, dr_pred, ds_dummy)

            # ---- carry to t-1 ----
            _vstore(sc + SB_DCARRY, dsprev, DS)


def aum_silence_bwd(streams, alpha, xw, k_rot, wpack, save, j_star, S_ckpt, dout, kappa,
                    forced=-1, no_op=False):
    """Same contract as kernels.metal.aum_silence_bwd: returns (demit (B,L,5443), dalpha,
    dxw, dkrot), all fp32. dout (B,L,2567) packs the incoming grads."""
    B, L = save.shape[:2]
    dev = save.device
    f32 = lambda x: x.contiguous().float()  # noqa: E731
    streams, alpha, xw, k_rot = f32(streams), f32(alpha), f32(xw), f32(k_rot)
    wpack, save, dout = f32(wpack), f32(save), f32(dout)
    assert dout.shape[-1] == DO.value and save.shape[-1] == SV.value
    demit = torch.empty(B, L, DE.value, device=dev, dtype=torch.float32)
    dalpha = torch.empty(B, L, NH.value, device=dev, dtype=torch.float32)
    dxw = torch.empty(B, L, NH.value * DH.value, device=dev, dtype=torch.float32)
    dkrot = torch.empty(B, L, NH.value * DH.value, device=dev, dtype=torch.float32)
    S_seg = torch.empty(B, 65, NH.value * DH.value * DH.value, device=dev,
                        dtype=torch.float32)
    dS = torch.zeros(B, NH.value * DH.value * DH.value, device=dev, dtype=torch.float32)
    scratch = torch.empty(B, SC_BWD.value, device=dev, dtype=torch.float32)
    _silence_bwd_kernel[(B,)](
        streams, alpha, xw.reshape(B, L, -1), k_rot.reshape(B, L, -1), wpack, save,
        j_star.contiguous(), S_ckpt.contiguous(), dout, demit, dalpha, dxw, dkrot,
        S_seg, dS, scratch, L, float(kappa),
        FORCED=int(forced) >= 0, NO_OP=bool(no_op), num_warps=8)
    return demit, dalpha, dxw, dkrot
