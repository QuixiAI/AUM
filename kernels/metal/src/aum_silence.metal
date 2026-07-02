#include <metal_stdlib>
using namespace metal;

// AUM-Ø fused sequential global block (v6 §5-§9; kernel-roadmap step 4) — FORWARD.
//
// Replaces AumBackbone._global_segment's per-token Python loop (launch-bound: ~50 tiny MPS ops
// x L tokens) with ONE kernel that marches the whole sequence. One threadgroup per batch row
// (the recurrence is sequential in t; batch rows are independent), 256 threads cooperating on
// each token's small matvecs / reductions. All token-parallel work (the g_t / m_t / s_t / phase
// column blocks of every concat-Linear, the rotation-ladder cos/sin) is precomputed OUTSIDE as
// input streams (aum_ssm/ops/metal/silence_flat.py defines the exact split and is the oracle).
//
// Geometry is HARDCODED to the reference config (checked at dispatch):
//   d_model D=512, d_sigma DS=128, d_mu DM=32, heads H=8, head dim DH=64, J=j_max=2, TPB=256.
// Scope: the standard training path (no ablations / top_gru / entropy_feature; halting via
// pre-drawn uniforms halt_u, or forced_depth >= 0).
//
// Per token t (matching silence_flat.flat_forward op-for-op):
//   S      <- alpha_t (.) S + x_t (x) k_t                    (evidence state, device memory)
//   r_pred  = S_prev . R(phi_{t-1}) W_qpred sigma            (predictive read; S_prev = pre-update)
//   g_hat   = W_P LN512(W_R r_pred + W_hyp sigma + c_phase)
//   e       = g_t - g_hat
//   mu      = sigmoid(c_mu + Wmu_e e + Wmu_s sigma);  e~ = mu (.) W_err e
//   sigma^0 = LN128(c_init + Wi_s sigma + Wi_et e~ + Wi_mu mu)
//   j = 0..J-1:  r_j = S . R(phi_t) W_qsig sigma^j
//                gate = sigmoid(c_gate + Wg_s sigma^j + Wg_et e~ + Wg_mu mu + Wg_r r_j)
//                cand = tanh   (c_cand + Wn_s sigma^j + Wn_et e~ + Wn_mu mu + Wn_r r_j)
//                sigma^{j+1} = LN128(sigma^j + gate (.) cand)
//   r_J     = S . R(phi_t) W_qsig sigma^J
//   E_j     = |precG(mu) (.) (c_pg - Q_G sigma^j)|^2 + |precR(mu) (.) (P_R r_j - Q_R sigma^j)|^2
//             + kappa |sigma^j - sigma_prev|^2
//   pi      = softplus(w_pi . tanh(c_press + wp_de*|e~| + wp_dsR*|P_R r_0 - Q_R sigma^0|))
//   p_j     = sigmoid(w_h2 . tanh(Wh1_s sigma^j + wh1_pi*pi + wh1_E*E_j)),  p_J = 1
//   w_j     = p_j prod_{i<j}(1-p_i);  j* = invCDF(w, u_t) or forced;  sigma <- sigma^{j*}
//
// Everything the losses / backward need is written to a per-token SAVE pack (incl. LayerNorm
// mean/rstd). o_stack, the prediction-head consumers and expected_J stay OUTSIDE as batched ops.

// ---------------- geometry ----------------
constant constexpr int D   = 512;   // d_model
constant constexpr int DS  = 128;   // d_sigma
constant constexpr int DM  = 32;    // d_mu
constant constexpr int NH  = 8;     // U heads
constant constexpr int DH  = 64;    // U head dim (Dv == Dk)
constant constexpr int PJ  = 2;     // j_max
constant constexpr int TPB = 256;   // threads per threadgroup
constant constexpr float LN_EPS = 1e-5f;

// stream pack per token: c_mu | c_init | c_gate | c_cand | c_pg | c_press | c_phase | g |
//                        cos_t | sin_t | cos_p | sin_p            (see silence_flat.py)
constant constexpr int SW_CMU    = 0;
constant constexpr int SW_CINIT  = SW_CMU + DM;
constant constexpr int SW_CGATE  = SW_CINIT + DS;
constant constexpr int SW_CCAND  = SW_CGATE + DS;
constant constexpr int SW_CPG    = SW_CCAND + DS;
constant constexpr int SW_CPRESS = SW_CPG + DS;
constant constexpr int SW_CPHASE = SW_CPRESS + DS;
constant constexpr int SW_G      = SW_CPHASE + D;
constant constexpr int SW_COST   = SW_G + D;
constant constexpr int SW_SINT   = SW_COST + NH * DH / 2;
constant constexpr int SW_COSP   = SW_SINT + NH * DH / 2;
constant constexpr int SW_SINP   = SW_COSP + NH * DH / 2;
constant constexpr int SW       = SW_SINP + NH * DH / 2;          // 2720

// save pack per token
constant constexpr int SV_GHAT   = 0;
constant constexpr int SV_MU     = SV_GHAT + D;
constant constexpr int SV_ETIL   = SV_MU + DM;
constant constexpr int SV_SIGST  = SV_ETIL + DM;                  // (J+1) x DS
constant constexpr int SV_RPRED  = SV_SIGST + (PJ + 1) * DS;
constant constexpr int SV_RST    = SV_RPRED + D;                  // (J+1) x D
constant constexpr int SV_E      = SV_RST + (PJ + 1) * D;
constant constexpr int SV_PI     = SV_E + (PJ + 1);
constant constexpr int SV_W      = SV_PI + 1;
constant constexpr int SV_SSTAR  = SV_W + (PJ + 1);
constant constexpr int SV_LN5    = SV_SSTAR + DS;                 // mean, rstd
constant constexpr int SV_LN1    = SV_LN5 + 2;                    // (mean, rstd) x (J+1) [sig0 + J revisions]
constant constexpr int SV       = SV_LN1 + 2 * (PJ + 1);          // 3152

// weight-pack offsets, in the ORDER build_silence_pack() concatenates (host mirrors this table)
constant constexpr int W_QPRED = 0;                               // (D, DS)
constant constexpr int W_R     = W_QPRED + D * DS;                // (D, D)
constant constexpr int W_HYP   = W_R + D * D;                     // (D, DS)
constant constexpr int W_P     = W_HYP + D * DS;                  // (D, D)
constant constexpr int W_LN5W  = W_P + D * D;                     // (D)
constant constexpr int W_LN5B  = W_LN5W + D;                      // (D)
constant constexpr int W_MUE   = W_LN5B + D;                      // (DM, D)
constant constexpr int W_MUS   = W_MUE + DM * D;                  // (DM, DS)
constant constexpr int W_ERR   = W_MUS + DM * DS;                 // (DM, D)
constant constexpr int W_IS    = W_ERR + DM * D;                  // (DS, DS)
constant constexpr int W_IET   = W_IS + DS * DS;                  // (DS, DM)
constant constexpr int W_IMU   = W_IET + DS * DM;                 // (DS, DM)
constant constexpr int W_LN1W  = W_IMU + DS * DM;                 // (DS)
constant constexpr int W_LN1B  = W_LN1W + DS;                     // (DS)
constant constexpr int W_QSIG  = W_LN1B + DS;                     // (D, DS)
constant constexpr int W_GS    = W_QSIG + D * DS;                 // (DS, DS)
constant constexpr int W_GET   = W_GS + DS * DS;                  // (DS, DM)
constant constexpr int W_GMU   = W_GET + DS * DM;                 // (DS, DM)
constant constexpr int W_GR    = W_GMU + DS * DM;                 // (DS, D)
constant constexpr int W_NS    = W_GR + DS * D;                   // (DS, DS)
constant constexpr int W_NET   = W_NS + DS * DS;                  // (DS, DM)
constant constexpr int W_NMU   = W_NET + DS * DM;                 // (DS, DM)
constant constexpr int W_NR    = W_NMU + DS * DM;                 // (DS, D)
constant constexpr int W_QG    = W_NR + DS * D;                   // (DS, DS)
constant constexpr int W_PRR   = W_QG + DS * DS;                  // (DS, D)   P_R
constant constexpr int W_QR    = W_PRR + DS * D;                  // (DS, DS)
constant constexpr int W_PCG   = W_QR + DS * DS;                  // (DS, DM)  prec_G
constant constexpr int W_PCR   = W_PCG + DS * DM;                 // (DS, DM)  prec_R
constant constexpr int W_WPI   = W_PCR + DS * DM;                 // (DS)
constant constexpr int W_WPDE  = W_WPI + DS;                      // (DS) pressure_in col de
constant constexpr int W_WPDS  = W_WPDE + DS;                     // (DS) pressure_in col dsR
constant constexpr int W_H1S   = W_WPDS + DS;                     // (64, DS)
constant constexpr int W_H1PI  = W_H1S + 64 * DS;                 // (64)
constant constexpr int W_H1E   = W_H1PI + 64;                     // (64)
constant constexpr int W_H2    = W_H1E + 64;                      // (64)
constant constexpr int W_TOTAL = W_H2 + 64;

// ---------------- cooperative helpers ----------------
// dst[o] (o in 0..OUT) accumulates W[o, :IN] . src[:IN]; W row-major (OUT, IN) at wp.
inline void matvec_add(threadgroup float *dst, device const float *wp,
                       threadgroup const float *src, int OUT, int IN, uint tid) {
    for (int o = (int)tid; o < OUT; o += TPB) {
        float acc = 0.0f;
        device const float *row = wp + (long)o * IN;
        for (int i = 0; i < IN; ++i) acc = fma(row[i], src[i], acc);
        dst[o] += acc;
    }
}

// threadgroup-wide sum of per-thread partials via simd + a small scratch array.
inline float block_sum(float v, threadgroup float *scratch, uint tid, uint simd_lane,
                       uint simd_id) {
    v = simd_sum(v);
    if (simd_lane == 0) scratch[simd_id] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (int i = 0; i < TPB / 32; ++i) s += scratch[i];
        scratch[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

// LayerNorm over vec[0..N) in threadgroup memory, affine from the weight pack; writes out
// (may alias vec) and the (mean, rstd) pair into save.
inline void layer_norm(threadgroup float *vec, threadgroup float *out, int N,
                       device const float *w, device const float *b,
                       device float *save_stats, threadgroup float *scratch,
                       uint tid, uint lane, uint sid) {
    float part = 0.0f, part2 = 0.0f;
    for (int i = (int)tid; i < N; i += TPB) { part += vec[i]; part2 += vec[i] * vec[i]; }
    float mean = block_sum(part, scratch, tid, lane, sid) / N;
    float var = block_sum(part2, scratch, tid, lane, sid) / N - mean * mean;
    float rstd = rsqrt(var + LN_EPS);
    for (int i = (int)tid; i < N; i += TPB) out[i] = (vec[i] - mean) * rstd * w[i] + b[i];
    if (tid == 0) { save_stats[0] = mean; save_stats[1] = rstd; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// r[h*DH + p] = sum_n S[h,p,n] * rot(q)[h,n]; q (D) in tg memory, rotated by (cos,sin) streams.
inline void rotate_query(threadgroup const float *q, device const float *cosv,
                         device const float *sinv, threadgroup float *q_rot, uint tid) {
    for (int i = (int)tid; i < D / 2; i += TPB) {           // i indexes (h, pair)
        float c = cosv[i], s = sinv[i];
        float x0 = q[2 * i], x1 = q[2 * i + 1];
        q_rot[2 * i] = x0 * c - x1 * s;
        q_rot[2 * i + 1] = x0 * s + x1 * c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void state_read(device const float *S, threadgroup const float *q_rot,
                       threadgroup float *r, uint tid) {
    for (int hp = (int)tid; hp < NH * DH; hp += TPB) {      // hp = h*DH + p
        const int h = hp / DH;
        device const float *Srow = S + (long)hp * DH;       // S[h, p, :]
        float acc = 0.0f;
        for (int n = 0; n < DH; ++n) acc = fma(Srow[n], q_rot[h * DH + n], acc);
        r[hp] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---------------- forward ----------------
kernel void aum_silence_fwd(
    device const float *streams  [[buffer(0)]],   // (B, L, SW)
    device const float *alpha    [[buffer(1)]],   // (B, L, NH)
    device const float *xw       [[buffer(2)]],   // (B, L, NH*DH)
    device const float *k_rot    [[buffer(3)]],   // (B, L, NH*DH)
    device const float *halt_u   [[buffer(4)]],   // (B, L)  (ignored when forced >= 0)
    device const float *wpack    [[buffer(5)]],   // (W_TOTAL)
    device       float *S        [[buffer(6)]],   // (B, NH, DH, DH) workspace, zeroed by host
    device       float *save     [[buffer(7)]],   // (B, L, SV)
    device       int   *j_star   [[buffer(8)]],   // (B, L)
    device       float *S_ckpt   [[buffer(9)]],   // (B, ceil(L/64), NH, DH, DH) — S_{t-1} at
    constant     uint  &L        [[buffer(10)]],  //   each segment start (backward replay)
    constant     float &kappa    [[buffer(11)]],
    constant     int   &forced   [[buffer(12)]],  // forced_depth, or -1
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sid  [[simdgroup_index_in_threadgroup]]) {

    const long b = tgid.z;
    device float *Sb = S + b * (long)(NH * DH * DH);

    threadgroup float sigma[DS];                 // the carried register sigma_{t-1}^{*}
    threadgroup float sig[(PJ + 1) * DS];        // sigma^0..sigma^J of the current token
    threadgroup float buf_d[D];                  // d_model scratch (pre / q / ...)
    threadgroup float buf_d2[D];                 // second d_model scratch (r / q_rot)
    threadgroup float e_vec[D];                  // e = g - g_hat
    threadgroup float mu[DM], etil[DM];
    threadgroup float small[DS];                 // DS-wide scratch (pre-acts, dG/dR, ...)
    threadgroup float small2[DS];
    threadgroup float muG[DS], muR[DS];
    threadgroup float hbuf[64];
    threadgroup float scratch[TPB / 32];
    threadgroup float scal[8];                   // [pi, E0, E1, E2, de, dsR, p_or_w tmp, u]

    for (int i = (int)tid; i < DS; i += TPB) sigma[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint nseg = (L + 63) / 64;
    for (uint t = 0; t < L; ++t) {
        device const float *st = streams + (b * L + t) * (long)SW;
        device float *sv = save + (b * L + t) * (long)SV;
        device const float *xt = xw + (b * L + t) * (long)(NH * DH);
        device const float *kt = k_rot + (b * L + t) * (long)(NH * DH);
        device const float *at = alpha + (b * L + t) * NH;

        if (t % 64 == 0) {                        // checkpoint S_{t-1} for the backward replay
            device float *ck = S_ckpt + (b * nseg + t / 64) * (long)(NH * DH * DH);
            for (int i = (int)tid; i < NH * DH * DH; i += TPB) ck[i] = Sb[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ---- predictive read against S_{t-1} (BEFORE the state update) ----
        for (int i = (int)tid; i < D; i += TPB) buf_d[i] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        matvec_add(buf_d, wpack + W_QPRED, sigma, D, DS, tid);          // q_pred = W_qpred sigma
        threadgroup_barrier(mem_flags::mem_threadgroup);
        rotate_query(buf_d, st + SW_COSP, st + SW_SINP, buf_d2, tid);   // R(phi_{t-1}) q
        state_read(Sb, buf_d2, e_vec, tid);                             // e_vec <- r_pred (temp)
        for (int i = (int)tid; i < D; i += TPB) sv[SV_RPRED + i] = e_vec[i];

        // ---- state update S = alpha (.) S + x (x) k ----
        for (int hp = (int)tid; hp < NH * DH; hp += TPB) {
            const int h = hp / DH, p = hp % DH;
            const float a = at[h], xv = xt[hp];
            device float *Srow = Sb + (long)hp * DH;
            device const float *kh = kt + h * DH;
            for (int n = 0; n < DH; ++n) Srow[n] = a * Srow[n] + xv * kh[n];
            (void)p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- g_hat = W_P LN512(W_R r_pred + W_hyp sigma + c_phase) ; e = g - g_hat ----
        for (int i = (int)tid; i < D; i += TPB) buf_d[i] = st[SW_CPHASE + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        matvec_add(buf_d, wpack + W_R, e_vec, D, D, tid);               // + W_R r_pred
        matvec_add(buf_d, wpack + W_HYP, sigma, D, DS, tid);            // + W_hyp sigma
        threadgroup_barrier(mem_flags::mem_threadgroup);
        layer_norm(buf_d, buf_d2, D, wpack + W_LN5W, wpack + W_LN5B, sv + SV_LN5,
                   scratch, tid, lane, sid);
        for (int i = (int)tid; i < D; i += TPB) buf_d[i] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        matvec_add(buf_d, wpack + W_P, buf_d2, D, D, tid);              // g_hat
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = (int)tid; i < D; i += TPB) {
            sv[SV_GHAT + i] = buf_d[i];
            e_vec[i] = st[SW_G + i] - buf_d[i];                         // e
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- mu, e_tilde ----
        for (int i = (int)tid; i < DM; i += TPB) small[i] = st[SW_CMU + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        matvec_add(small, wpack + W_MUE, e_vec, DM, D, tid);
        matvec_add(small, wpack + W_MUS, sigma, DM, DS, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = (int)tid; i < DM; i += TPB) {
            mu[i] = 1.0f / (1.0f + exp(-small[i]));
            small2[i] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        matvec_add(small2, wpack + W_ERR, e_vec, DM, D, tid);           // W_err e
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = (int)tid; i < DM; i += TPB) {
            etil[i] = mu[i] * small2[i];
            sv[SV_MU + i] = mu[i];
            sv[SV_ETIL + i] = etil[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- sigma^0 ----
        for (int i = (int)tid; i < DS; i += TPB) small[i] = st[SW_CINIT + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        matvec_add(small, wpack + W_IS, sigma, DS, DS, tid);
        matvec_add(small, wpack + W_IET, etil, DS, DM, tid);
        matvec_add(small, wpack + W_IMU, mu, DS, DM, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        layer_norm(small, sig, DS, wpack + W_LN1W, wpack + W_LN1B, sv + SV_LN1,
                   scratch, tid, lane, sid);

        // ---- revision loop: j = 0..J-1 revise, plus the final read at sigma^J ----
        for (int j = 0; j <= PJ; ++j) {
            threadgroup float *sj = sig + j * DS;
            // r_j = S . R(phi_t) W_qsig sigma^j
            for (int i = (int)tid; i < D; i += TPB) buf_d[i] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            matvec_add(buf_d, wpack + W_QSIG, sj, D, DS, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            rotate_query(buf_d, st + SW_COST, st + SW_SINT, buf_d2, tid);
            state_read(Sb, buf_d2, buf_d, tid);                         // buf_d <- r_j
            for (int i = (int)tid; i < D; i += TPB) sv[SV_RST + j * D + i] = buf_d[i];
            for (int i = (int)tid; i < DS; i += TPB) sv[SV_SIGST + j * DS + i] = sj[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (j == PJ) break;
            // gate / cand
            for (int i = (int)tid; i < DS; i += TPB) {
                small[i] = st[SW_CGATE + i];
                small2[i] = st[SW_CCAND + i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            matvec_add(small, wpack + W_GS, sj, DS, DS, tid);
            matvec_add(small, wpack + W_GET, etil, DS, DM, tid);
            matvec_add(small, wpack + W_GMU, mu, DS, DM, tid);
            matvec_add(small, wpack + W_GR, buf_d, DS, D, tid);
            matvec_add(small2, wpack + W_NS, sj, DS, DS, tid);
            matvec_add(small2, wpack + W_NET, etil, DS, DM, tid);
            matvec_add(small2, wpack + W_NMU, mu, DS, DM, tid);
            matvec_add(small2, wpack + W_NR, buf_d, DS, D, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = (int)tid; i < DS; i += TPB) {
                float gate = 1.0f / (1.0f + exp(-small[i]));
                float cand = tanh(small2[i]);
                small[i] = sj[i] + gate * cand;                         // pre-LN
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            layer_norm(small, sig + (j + 1) * DS, DS, wpack + W_LN1W, wpack + W_LN1B,
                       sv + SV_LN1 + 2 * (j + 1), scratch, tid, lane, sid);
        }

        // ---- precision-weighted consistency E_j; pressure pi ----
        for (int i = (int)tid; i < DS; i += TPB) { muG[i] = 0.0f; muR[i] = 0.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        matvec_add(muG, wpack + W_PCG, mu, DS, DM, tid);
        matvec_add(muR, wpack + W_PCR, mu, DS, DM, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j <= PJ; ++j) {
            threadgroup float *sj = sig + j * DS;
            device const float *rj = sv + SV_RST + j * D;               // saved r_j (device)
            // small = c_pg - Q_G sigma^j ; small2 = P_R r_j - Q_R sigma^j
            for (int i = (int)tid; i < DS; i += TPB) { small[i] = 0.0f; small2[i] = 0.0f; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            matvec_add(small, wpack + W_QG, sj, DS, DS, tid);
            matvec_add(small2, wpack + W_QR, sj, DS, DS, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float part = 0.0f;
            for (int i = (int)tid; i < DS; i += TPB) {
                float dG = st[SW_CPG + i] - small[i];
                float pr = 0.0f;
                device const float *row = wpack + W_PRR + (long)i * D;
                for (int n = 0; n < D; ++n) pr = fma(row[n], rj[n], pr);
                float dR = pr - small2[i];
                float ds = sj[i] - sigma[i];
                part += muG[i] * muG[i] * dG * dG + muR[i] * muR[i] * dR * dR
                        + kappa * ds * ds;
                if (j == 0) small[i] = dR;                              // keep dR_0 for dsR
            }
            float Ej = block_sum(part, scratch, tid, lane, sid);
            if (tid == 0) { scal[1 + j] = Ej; sv[SV_E + j] = Ej; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (j == 0) {                                               // dsR = |dR_0|
                float p2 = 0.0f;
                for (int i = (int)tid; i < DS; i += TPB) p2 += small[i] * small[i];
                float n2 = block_sum(p2, scratch, tid, lane, sid);
                if (tid == 0) scal[5] = sqrt(n2);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        {   // de = |e_tilde| ; pi = softplus(w_pi . tanh(c_press + wp_de*de + wp_dsR*dsR))
            float p2 = 0.0f;
            for (int i = (int)tid; i < DM; i += TPB) p2 += etil[i] * etil[i];
            float n2 = block_sum(p2, scratch, tid, lane, sid);
            if (tid == 0) scal[4] = sqrt(n2);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float part = 0.0f;
            for (int i = (int)tid; i < DS; i += TPB) {
                float pre = st[SW_CPRESS + i] + wpack[W_WPDE + i] * scal[4]
                            + wpack[W_WPDS + i] * scal[5];
                part += wpack[W_WPI + i] * tanh(pre);
            }
            float z = block_sum(part, scratch, tid, lane, sid);
            if (tid == 0) {
                float pi = z > 20.0f ? z : log(1.0f + exp(z));
                scal[0] = pi;
                sv[SV_PI] = pi;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ---- halting: p_j -> w_j -> j* ; carry sigma <- sigma^{j*} ----
        for (int j = 0; j < PJ; ++j) {
            for (int i = (int)tid; i < 64; i += TPB) hbuf[i] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            matvec_add(hbuf, wpack + W_H1S, sig + j * DS, 64, DS, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float part = 0.0f;
            for (int i = (int)tid; i < 64; i += TPB) {
                float h = tanh(hbuf[i] + wpack[W_H1PI + i] * scal[0]
                               + wpack[W_H1E + i] * scal[1 + j]);
                part += wpack[W_H2 + i] * h;
            }
            float z = block_sum(part, scratch, tid, lane, sid);
            if (tid == 0) scal[6 + j] = 1.0f / (1.0f + exp(-z));        // p_j, j in {0, 1}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            const float p0 = scal[6], p1 = scal[7];
            float w0 = p0, w1 = p1 * (1.0f - p0), w2 = (1.0f - p0) * (1.0f - p1);
            sv[SV_W + 0] = w0; sv[SV_W + 1] = w1; sv[SV_W + 2] = w2;
            int js;
            if (forced >= 0) {
                js = forced;
                sv[SV_W + 0] = js == 0 ? 1.0f : 0.0f;
                sv[SV_W + 1] = js == 1 ? 1.0f : 0.0f;
                sv[SV_W + 2] = js == 2 ? 1.0f : 0.0f;
            } else {
                const float u = halt_u[b * L + t];
                js = (w0 >= u) ? 0 : ((w0 + w1 >= u) ? 1 : 2);
            }
            j_star[b * L + t] = js;
            scal[3] = (float)js;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int js = (int)scal[3];
        for (int i = (int)tid; i < DS; i += TPB) {
            sigma[i] = sig[js * DS + i];
            sv[SV_SSTAR + i] = sigma[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
