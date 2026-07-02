# Metal path for the fused sequential global block (v6 §5-§9; kernel-roadmap step 4).
#
# Pack builders mirror the layouts hardcoded in kernels/metal/src/aum_silence.metal:
#   * build_weight_pack : one flat fp32 tensor, torch.cat of the module's weight column blocks in
#     the kernel's W_* offset order. Built INSIDE the autograd graph, so the backward kernel's
#     d(wpack) routes to the parameters through this cat.
#   * build_stream_pack : (B, L, SW) — the token-parallel precompute streams + g + cos/sin.
#
# unpack_save() reslices the kernel's per-token save pack into the named trajectories.

import torch
import torch.nn.functional as F

from aum_ssm.modules.silence import _phase_embed
from aum_ssm.ops.metal.silence_flat import _cols

SW, SV, J = 2720, 3151, 2                       # must match aum_silence.metal


def build_weight_pack(silence):
    """Flat fp32 weight pack in the kernel's offset order (differentiable cat)."""
    d, s, m = silence.d_model, silence.d_sigma, silence.d_mu
    _, mu_e, mu_s, _ = _cols(silence.modulate.in_proj_mu.weight, d, d, s, m)
    i_s, _, i_et, i_mu = _cols(silence.register.init_proj.weight, s, d, m, m)
    g_s, _, g_et, g_mu, g_r = _cols(silence.register.update_gate.weight, s, d, m, m, d)
    n_s, _, n_et, n_mu, n_r = _cols(silence.register.update_cand.weight, s, d, m, m, d)
    wp = silence.pressure_halt.pressure_in.weight
    h1_s, h1_pi, h1_e = _cols(silence.pressure_halt.halt_1.weight, s, 1, 1)
    parts = [
        silence.predict.query_proj.weight,      # W_QPRED (D, DS)
        silence.predict.read_proj.weight,       # W_R     (D, D)
        silence.predict.hyp_proj.weight,        # W_HYP   (D, DS)
        silence.predict.out_proj.weight,        # W_P     (D, D)
        silence.predict.norm.weight,            # LN5 w
        silence.predict.norm.bias,              # LN5 b
        mu_e, mu_s,                             # W_MUE, W_MUS
        silence.modulate.err_proj.weight,       # W_ERR
        i_s, i_et, i_mu,                        # W_IS, W_IET, W_IMU
        silence.register.norm.weight,           # LN1 w
        silence.register.norm.bias,             # LN1 b
        silence.register.read_proj.weight,      # W_QSIG (D, DS)
        g_s, g_et, g_mu, g_r,                   # W_G*
        n_s, n_et, n_mu, n_r,                   # W_N*
        silence.consistency.Q_G.weight,         # W_QG
        silence.consistency.P_R.weight,         # W_PRR
        silence.consistency.Q_R.weight,         # W_QR
        silence.consistency.prec_G.weight,      # W_PCG
        silence.consistency.prec_R.weight,      # W_PCR
        silence.pressure_halt.pressure_out,     # W_WPI
        wp[:, d], wp[:, d + 1],                 # W_WPDE, W_WPDS (scalar feature columns)
        h1_s, h1_pi[:, 0], h1_e[:, 0],          # W_H1S, W_H1PI, W_H1E
        silence.pressure_halt.halt_2.weight[0],  # W_H2
    ]
    return torch.cat([p.reshape(-1) for p in parts]).float()


def build_stream_pack(silence, g, phi, m_t, s_t, freqs):
    """(B, L, SW): the token-parallel streams, in the kernel's SW_* order (differentiable)."""
    B, L, H = phi.shape
    d, s, m = silence.d_model, silence.d_sigma, silence.d_mu
    Wmu_g, _, _, Wmu_m = _cols(silence.modulate.in_proj_mu.weight, d, d, s, m)
    Wi_g = _cols(silence.register.init_proj.weight, s, d, m, m)[1]
    Wg_g = _cols(silence.register.update_gate.weight, s, d, m, m, d)[1]
    Wn_g = _cols(silence.register.update_cand.weight, s, d, m, m, d)[1]
    Wp = silence.pressure_halt.pressure_in.weight
    phi_prev = torch.cat([phi.new_zeros(B, 1, H), phi[:, :-1]], dim=1)
    ang_t = (phi.unsqueeze(-1) * freqs).reshape(B, L, -1)
    ang_p = (phi_prev.unsqueeze(-1) * freqs).reshape(B, L, -1)
    return torch.cat([
        F.linear(g, Wmu_g) + F.linear(m_t, Wmu_m),                    # c_mu     (32)
        F.linear(g, Wi_g),                                            # c_init   (128)
        F.linear(g, Wg_g),                                            # c_gate   (128)
        F.linear(g, Wn_g),                                            # c_cand   (128)
        silence.consistency.P_G(g),                                   # c_pg     (128)
        F.linear(g, Wp[:, :d]) + Wp[:, d + 2] * s_t,                  # c_press  (128)
        silence.predict.phase_proj(_phase_embed(phi_prev, silence.d_phase)),  # c_phase (512)
        g,                                                            # g        (512)
        torch.cos(ang_t), torch.sin(ang_t),                           # cos/sin_t (256 each)
        torch.cos(ang_p), torch.sin(ang_p),                           # cos/sin_p (256 each)
    ], dim=-1).float()


def unpack_save(save):
    """Reslice the (B, L, SV) save pack into named fp32 trajectories."""
    D, DS, DM = 512, 128, 32
    off, out = 0, {}
    for name, width in [("g_hat", D), ("mu", DM), ("e_tilde", DM),
                        ("sigma_stack", (J + 1) * DS), ("r_pred", D), ("r_stack", (J + 1) * D),
                        ("E", J + 1), ("pi", 1), ("w", J + 1), ("sigma_star", DS),
                        ("ln5", 2), ("ln1", 2 * (J + 1))]:
        out[name] = save[..., off:off + width]
        off += width
    assert off == SV, off
    B, L = save.shape[:2]
    out["sigma_stack"] = out["sigma_stack"].reshape(B, L, J + 1, DS)
    out["r_stack"] = out["r_stack"].reshape(B, L, J + 1, D)
    out["pi"] = out["pi"].squeeze(-1)
    return out


def silence_fwd_metal(silence, g, phi, m_t, s_t, alpha, xw, k_rot, freqs,
                      halt_u=None, forced_depth=None):
    """Run the fused forward (no autograd; tests/benchmarks).
    Returns (unpacked trajectories dict, j_star, save, packs)."""
    import kernels.metal as km
    streams = build_stream_pack(silence, g, phi, m_t, s_t, freqs)
    wpack = build_weight_pack(silence)
    B, L = g.shape[:2]
    hu = halt_u if halt_u is not None else g.new_zeros(B, L)
    forced = -1 if forced_depth is None else int(forced_depth)
    save, j_star, S_final, S_ckpt = km.aum_silence_fwd(
        streams.detach(), alpha.detach().float(),
        xw.detach().float().reshape(B, L, -1), k_rot.detach().float().reshape(B, L, -1),
        hu.float(), wpack.detach(), silence.kappa, forced)
    out = unpack_save(save)
    return out, j_star, save, (streams, wpack, S_ckpt)


# ------------------------------ backward assembly ------------------------------
# demit pack offsets (must match aum_silence.metal DE_*)
D, DS, DM = 512, 128, 32

# stream-pack slice offsets (must match aum_silence.metal SW_*)
_SW = {}
_o = 0
for _n, _w2 in [("c_mu", DM), ("c_init", DS), ("c_gate", DS), ("c_cand", DS), ("c_pg", DS),
                ("c_press", DS), ("c_phase", D), ("g", D), ("cos_t", 256), ("sin_t", 256),
                ("cos_p", 256), ("sin_p", 256)]:
    _SW[_n] = (_o, _o + _w2)
    _o += _w2
assert _o == SW


def _weight_offsets():
    """{name: (start, end, shape)} for the flat weight pack (must match aum_silence.metal W_*)."""
    table = [("W_QPRED", (D, DS)), ("W_R", (D, D)), ("W_HYP", (D, DS)), ("W_P", (D, D)),
             ("LN5W", (D,)), ("LN5B", (D,)), ("W_MUE", (DM, D)), ("W_MUS", (DM, DS)),
             ("W_ERR", (DM, D)), ("W_IS", (DS, DS)), ("W_IET", (DS, DM)), ("W_IMU", (DS, DM)),
             ("LN1W", (DS,)), ("LN1B", (DS,)), ("W_QSIG", (D, DS)),
             ("W_GS", (DS, DS)), ("W_GET", (DS, DM)), ("W_GMU", (DS, DM)), ("W_GR", (DS, D)),
             ("W_NS", (DS, DS)), ("W_NET", (DS, DM)), ("W_NMU", (DS, DM)), ("W_NR", (DS, D)),
             ("W_QG", (DS, DS)), ("W_PRR", (DS, D)), ("W_QR", (DS, DS)),
             ("W_PCG", (DS, DM)), ("W_PCR", (DS, DM)), ("W_WPI", (DS,)), ("W_WPDE", (DS,)),
             ("W_WPDS", (DS,)), ("W_H1S", (64, DS)), ("W_H1PI", (64,)), ("W_H1E", (64,)),
             ("W_H2", (64,))]
    out, off = {}, 0
    for name, shape in table:
        n = 1
        for s in shape:
            n *= s
        out[name] = (off, off + n, shape)
        off += n
    return out
_DE = {}
_off = 0
for _n, _w in [("pre", D), ("ghat", D), ("zmu", DM), ("ett", DM), ("zi", DS),
               ("ds", 3 * DS), ("zg", 2 * DS), ("zn", 2 * DS), ("qp", D), ("qs", 3 * D),
               ("dg", 3 * DS), ("drr", 3 * DS), ("mug", DS), ("mur", DS), ("ppi", DS),
               ("zpi", 1), ("h", 2 * 64), ("zh", 2)]:
    _DE[_n] = (_off, _off + _w)
    _off += _w
DE_W = _off                                           # 5443


def _sl(t, name):
    a, b = _DE[name]
    return t[..., a:b]


def _rot_pair_grads(q, dq, cos, sin):
    """Given the UNROTATED query q and its grad dq, return (dcos, dsin) for qr = R(phi) q.
    q/dq (T, D); cos/sin (T, D/2). dqr = R(phi) dq (rotating a gradient of the rotated vector
    back forward), then dcos_i = q0*dqr0 + q1*dqr1, dsin_i = -q1*dqr0 + q0*dqr1."""
    q0, q1 = q[..., 0::2], q[..., 1::2]
    g0, g1 = dq[..., 0::2], dq[..., 1::2]
    dqr0 = cos * g0 - sin * g1
    dqr1 = sin * g0 + cos * g1
    return q0 * dqr0 + q1 * dqr1, -q1 * dqr0 + q0 * dqr1


class _SilenceFused(torch.autograd.Function):
    """The fused sequential global block: fwd/bwd Metal kernels + host GEMM grad assembly.

    forward returns the SAVE pack; consumers slice it (unpack_save), so autograd delivers the
    incoming grads per-slot in grad_save. Supported grad slots: sigma_stack, g_hat, r_stack, E,
    pi, w, sigma_star (grads landing in mu / e_tilde / r_pred / LN-stat slots are NOT
    propagated — no training loss consumes those)."""

    @staticmethod
    def forward(ctx, streams, wpack, alpha, xwf, krotf, halt_u, kappa, forced, no_op,
                halt_mode, delta):
        import kernels.metal as km
        save, j_star, _S, S_ckpt = km.aum_silence_fwd(
            streams.detach(), alpha.detach(), xwf.detach(), krotf.detach(), halt_u,
            wpack.detach(), kappa, forced, no_op, halt_mode, delta)
        ctx.save_for_backward(streams, wpack, alpha, xwf, krotf, save, j_star, S_ckpt)
        ctx.meta = (kappa, forced, no_op)
        return save, j_star

    @staticmethod
    def backward(ctx, gsave, _gj):
        import kernels.metal as km
        streams, wpack, alpha, xwf, krotf, save, j_star, S_ckpt = ctx.saved_tensors
        kappa, forced, no_op = ctx.meta
        B, L = save.shape[:2]
        T = B * L
        out = unpack_save(save)
        gs = unpack_save(gsave)

        # dout pack: d sigma_stack | d g_hat | d r_stack | d E | d pi | d w | d sigma_star
        dout = torch.cat([
            gs["sigma_stack"].reshape(B, L, -1), gs["g_hat"], gs["r_stack"].reshape(B, L, -1),
            gs["E"], gs["pi"].unsqueeze(-1), gs["w"], gs["sigma_star"],
        ], dim=-1).contiguous()
        demit, dalpha, dxw, dkrot = km.aum_silence_bwd(
            streams, alpha, xwf, krotf, wpack, save, j_star, S_ckpt, dout, kappa,
            forced, no_op)

        # ---- flatten to (T, .) for the GEMM assembly ----
        de = demit.reshape(T, -1)
        wof = _weight_offsets()
        W = lambda n: wpack[wof[n][0]:wof[n][1]].view(*wof[n][2])  # noqa: E731
        sig = out["sigma_stack"].reshape(T, 3, DS)
        rst = out["r_stack"].reshape(T, 3, D)
        r_pred = out["r_pred"].reshape(T, D)
        g_hat = out["g_hat"].reshape(T, D)
        mu = out["mu"].reshape(T, DM)
        etil = out["e_tilde"].reshape(T, DM)
        E = out["E"].reshape(T, 3)
        pi = out["pi"].reshape(T)
        ln5 = out["ln5"].reshape(T, 2)
        ln1 = out["ln1"].reshape(T, 3, 2)
        sstar = out["sigma_star"]
        sprev = torch.cat([sstar.new_zeros(B, 1, DS), sstar[:, :-1]], dim=1).reshape(T, DS)
        stfl = streams.reshape(T, SW)
        c_press = stfl[:, _SW["c_press"][0]:_SW["c_press"][1]]
        c_phase = stfl[:, _SW["c_phase"][0]:_SW["c_phase"][1]]
        g_in = stfl[:, _SW["g"][0]:_SW["g"][1]]
        cos_t = stfl[:, _SW["cos_t"][0]:_SW["cos_t"][1]]
        sin_t = stfl[:, _SW["sin_t"][0]:_SW["sin_t"][1]]
        cos_p = stfl[:, _SW["cos_p"][0]:_SW["cos_p"][1]]
        sin_p = stfl[:, _SW["sin_p"][0]:_SW["sin_p"][1]]
        e_vec = g_in - g_hat
        op = lambda a, b: a.transpose(0, 1) @ b                     # noqa: E731  (out, in) grads

        dz_mu = _sl(de, "zmu")
        dett = _sl(de, "ett")
        dz_i = _sl(de, "zi")
        dds = _sl(de, "ds").reshape(T, 3, DS)                       # LN1 output grads (dy)
        dzg = _sl(de, "zg").reshape(T, 2, DS)
        dzn = _sl(de, "zn").reshape(T, 2, DS)
        dqp = _sl(de, "qp")
        dqs = _sl(de, "qs").reshape(T, 3, D)
        ddG = _sl(de, "dg").reshape(T, 3, DS)
        ddR = _sl(de, "drr").reshape(T, 3, DS)
        dmuG = _sl(de, "mug")
        dmuR = _sl(de, "mur")
        dppi = _sl(de, "ppi")
        dzpi = _sl(de, "zpi")[:, 0]
        dh = _sl(de, "h").reshape(T, 2, 64)
        dzh = _sl(de, "zh")
        dpre = _sl(de, "pre")
        dght = _sl(de, "ghat")

        # ---- recomputes needed for weight grads (all batched GEMMs, fp32) ----
        pre5 = r_pred @ W("W_R").T + sprev @ W("W_HYP").T + c_phase
        xhat5 = (pre5 - ln5[:, :1]) * ln5[:, 1:]
        ln5_out = xhat5 * W("LN5W") + W("LN5B")
        dy5 = dght @ W("W_P")                                       # LN5 output grad
        zi_v = (stfl[:, _SW["c_init"][0]:_SW["c_init"][1]] + sprev @ W("W_IS").T
                + etil @ W("W_IET").T + mu @ W("W_IMU").T)
        xhat1 = [(zi_v - ln1[:, 0, :1]) * ln1[:, 0, 1:]]
        for j in range(2):                                          # revise pre-LN inputs
            zg = (stfl[:, _SW["c_gate"][0]:_SW["c_gate"][1]] + sig[:, j] @ W("W_GS").T
                  + etil @ W("W_GET").T + mu @ W("W_GMU").T + rst[:, j] @ W("W_GR").T)
            zn = (stfl[:, _SW["c_cand"][0]:_SW["c_cand"][1]] + sig[:, j] @ W("W_NS").T
                  + etil @ W("W_NET").T + mu @ W("W_NMU").T + rst[:, j] @ W("W_NR").T)
            v = sig[:, j] + torch.sigmoid(zg) * torch.tanh(zn)
            xhat1.append((v - ln1[:, j + 1, :1]) * ln1[:, j + 1, 1:])
        de_n = etil.norm(dim=-1)
        dR0 = rst[:, 0] @ W("W_PRR").T - sig[:, 0] @ W("W_QR").T
        dsR = dR0.norm(dim=-1)
        pre_pi = c_press + W("W_WPDE") * de_n[:, None] + W("W_WPDS") * dsR[:, None]
        h_pre = (sig[:, :2] @ W("W_H1S").T
                 + W("W_H1PI") * pi[:, None, None] + W("W_H1E") * E[:, :2, None])
        h_v = torch.tanh(h_pre)                                     # (T, 2, 64)

        # ---- weight grads, in build_weight_pack order ----
        # NOTE plain mm, never einsum: torch.einsum("tjo,tji->oi") on MPS materializes the
        # broadcast product ((T,J,O,I) — tens of GB at B=8) instead of contracting.
        dj = lambda a, b: (a.reshape(-1, a.shape[-1]).T             # noqa: E731  sum over (t, j)
                           @ b.reshape(-1, b.shape[-1]))
        parts = [
            op(dqp, sprev),                                         # W_QPRED
            op(dpre, r_pred),                                       # W_R
            op(dpre, sprev),                                        # W_HYP
            op(dght, ln5_out),                                      # W_P
            (dy5 * xhat5).sum(0),                                   # LN5 w
            dy5.sum(0),                                             # LN5 b
            op(dz_mu, e_vec),                                       # W_MUE
            op(dz_mu, sprev),                                       # W_MUS
            op(mu * dett, e_vec),                                   # W_ERR
            op(dz_i, sprev), op(dz_i, etil), op(dz_i, mu),          # W_IS, W_IET, W_IMU
            sum((dds[:, j] * xhat1[j]).sum(0) for j in range(3)),   # LN1 w
            dds.sum(dim=(0, 1)),                                    # LN1 b
            dj(dqs, sig),                                           # W_QSIG
            dj(dzg, sig[:, :2]), dj(dzg, etil[:, None].expand(T, 2, DM)),
            dj(dzg, mu[:, None].expand(T, 2, DM)), dj(dzg, rst[:, :2]),   # W_G*
            dj(dzn, sig[:, :2]), dj(dzn, etil[:, None].expand(T, 2, DM)),
            dj(dzn, mu[:, None].expand(T, 2, DM)), dj(dzn, rst[:, :2]),   # W_N*
            -dj(ddG, sig),                                          # Q_G
            dj(ddR, rst),                                           # P_R
            -dj(ddR, sig),                                          # Q_R
            op(dmuG, mu), op(dmuR, mu),                             # prec_G, prec_R
            (torch.tanh(pre_pi) * dzpi[:, None]).sum(0),            # w_pi
            (dppi * de_n[:, None]).sum(0),                          # wpde
            (dppi * dsR[:, None]).sum(0),                           # wpds
            dj(dh, sig[:, :2]),                                     # W_H1S
            (dh * pi[:, None, None]).sum(dim=(0, 1)),               # wh1_pi
            (dh * E[:, :2, None]).sum(dim=(0, 1)),                  # wh1_E
            (h_v * dzh[:, :, None]).sum(dim=(0, 1)),                # w_h2
        ]
        dwpack = torch.cat([p.reshape(-1) for p in parts])

        # ---- stream grads, in SW order ----
        dcos_t, dsin_t = torch.zeros_like(cos_t), torch.zeros_like(sin_t)
        for j in range(3):
            qj = sig[:, j] @ W("W_QSIG").T
            dc, ds_ = _rot_pair_grads(qj, dqs[:, j], cos_t, sin_t)
            dcos_t += dc
            dsin_t += ds_
        qpred = sprev @ W("W_QPRED").T
        dcos_p, dsin_p = _rot_pair_grads(qpred, dqp, cos_p, sin_p)
        de_g = gs["g_hat"].reshape(T, D) - dght                     # de = dghat_in - dGhat_tot
        dstreams = torch.cat([
            dz_mu, dz_i, dzg.sum(1), dzn.sum(1), ddG.sum(1), dppi, dpre, de_g,
            dcos_t, dsin_t, dcos_p, dsin_p,
        ], dim=-1).reshape(B, L, SW)

        return (dstreams, dwpack, dalpha, dxw, dkrot, None, None, None, None, None, None)


def silence_fused(silence, g, phi, m_t, s_t, alpha, xw, k_rot, freqs, halt_u=None,
                  forced_depth=None, no_op=False):
    """The fused global recurrence, end to end: packs -> Metal fwd (+autograd bwd) -> SilenceAux.

    Drop-in for the backbone's per-token segment loop on the standard path (no ablations other
    than no_op, no top_gru/entropy_feature/pi_trigger, reference geometry). Halting: j* ~
    Categorical(w) via halt_u when given, else the inference rule min{j: p_j >= delta};
    forced_depth pins one-hot w. Returns (o_t, SilenceAux) shaped exactly like SilenceBlock.
    """
    from aum_ssm.modules.silence import SilenceAux

    B, L = g.shape[:2]
    streams = build_stream_pack(silence, g, phi, m_t, s_t, freqs)
    wpack = build_weight_pack(silence)
    hu = halt_u.float() if halt_u is not None else g.new_zeros(B, L)
    forced = -1 if forced_depth is None else int(forced_depth)
    halt_mode = 0 if (halt_u is not None or forced >= 0) else 2
    save, j_star = _SilenceFused.apply(
        streams, wpack, alpha.float().reshape(B, L, -1), xw.float().reshape(B, L, -1),
        k_rot.float().reshape(B, L, -1), hu, float(silence.kappa), forced, int(no_op),
        halt_mode, float(silence.halt_delta))
    out = unpack_save(save)
    j_star = j_star.long()

    sigma_stack = out["sigma_stack"]                       # (B, L, J+1, DS) — the REAL trajectory
    out_stack = sigma_stack
    if no_op:                                              # §14: revision discarded in the output
        out_stack = sigma_stack[:, :, :1].expand_as(sigma_stack)
    o_stack = silence._output(g.unsqueeze(-2), out_stack)  # (B, L, J+1, d_model)
    idx = (j_star if not no_op else torch.zeros_like(j_star))[..., None, None]
    o_t = o_stack.gather(-2, idx.expand(B, L, 1, silence.d_model)).squeeze(-2)
    w = out["w"]
    expected_J = (w * torch.arange(silence.j_max + 1, device=w.device, dtype=w.dtype)).sum(-1)
    aux = SilenceAux(
        g=g, g_hat=out["g_hat"], e=g - out["g_hat"], mu=out["mu"], e_tilde=out["e_tilde"],
        sigma0=sigma_stack[:, :, 0], sigma_traj=list(sigma_stack.unbind(2)),
        r_traj=list(out["r_stack"].unbind(2)), E_traj=out["E"], pi=out["pi"], w=w,
        expected_J=expected_J, o_stack=o_stack, j_star=j_star, sigma_star=out["sigma_star"],
        phi=phi)
    return o_t, aux
