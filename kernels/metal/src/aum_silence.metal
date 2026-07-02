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
    constant     int   &no_op    [[buffer(13)]],  // §14 no_op ablation (stage 1): carry sigma^0
    constant     int   &haltmode [[buffer(14)]],  // 0: j* ~ invCDF(halt_u); 2: min{j: p_j>=delta}
    constant     float &delta    [[buffer(15)]],  // inference halting threshold (§8)
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
            } else if (haltmode == 2) {                // inference: first p_j >= delta (p_J = 1)
                js = (p0 >= delta) ? 0 : ((p1 >= delta) ? 1 : 2);
            } else {
                const float u = halt_u[b * L + t];
                js = (w0 >= u) ? 0 : ((w0 + w1 >= u) ? 1 : 2);
            }
            j_star[b * L + t] = js;
            scal[3] = (float)js;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int js = no_op ? 0 : (int)scal[3];   // no_op: revision discarded, sigma^0 carried
        for (int i = (int)tid; i < DS; i += TPB) {
            sigma[i] = sig[js * DS + i];
            sv[SV_SSTAR + i] = sigma[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------
// BACKWARD — reverse march with exact per-token replay.
//
// Design: the kernel walks t = L-1 .. 0 carrying the two sequential gradients (d_sigma through
// the register carry, dS through the evidence-state chain) and REPLAYS each token's small
// nonlinear pipeline from the forward save pack (recomputing matvec pre-activations; LayerNorm
// via saved mean/rstd). S_{t-1}/S_t come from a per-64-token segment replay off the forward's
// S checkpoints — never 1/alpha reconstruction. Instead of accumulating weight gradients
// in-kernel (an outer product per matvec per token — bandwidth-catastrophic), the kernel EMITS
// the per-token d-vectors (dz of every gate/projection, dq of every read, LN output grads) into
// a demit pack; the host then forms every weight and stream gradient as ONE batched GEMM over
// (B*L) rows (silence_metal.py::silence_bwd_assemble). d_alpha / d_x / d_k are computed
// in-kernel (they need dS_t and S_{t-1}).
//
// Incoming grads (dout pack): d sigma_stack | d g_hat | d r_stack | d E | d pi | d w.
// (sigma_star / expected_J / o_stack / prediction-head grads arrive through those slots via the
// host-side consumers.) When forced >= 0 the halting weights were one-hot (no p-path gradient),
// so the halting backward is skipped — matching F.one_hot in the module.

constant constexpr int DO_SIG  = 0;                    // (J+1) x DS
constant constexpr int DO_GHAT = DO_SIG + (PJ + 1) * DS;
constant constexpr int DO_RST  = DO_GHAT + D;          // (J+1) x D
constant constexpr int DO_E    = DO_RST + (PJ + 1) * D;
constant constexpr int DO_PI   = DO_E + (PJ + 1);
constant constexpr int DO_W    = DO_PI + 1;
constant constexpr int DO_SST  = DO_W + (PJ + 1);      // d sigma_star (routes to sigma^{j*})
constant constexpr int DO      = DO_SST + DS;          // 2567

constant constexpr int DE_PRE  = 0;                    // dpre (LN5 input grad)        (D)
constant constexpr int DE_GHT  = DE_PRE + D;           // dGhat_tot                    (D)
constant constexpr int DE_ZMU  = DE_GHT + D;           // dz_mu                        (DM)
constant constexpr int DE_ETT  = DE_ZMU + DM;          // d e_tilde (collected)        (DM)
constant constexpr int DE_ZI   = DE_ETT + DM;          // dz_init                      (DS)
constant constexpr int DE_DS0  = DE_ZI + DS;           // d sigma^j totals (LN1 outs)  (DS)x3
constant constexpr int DE_ZG0  = DE_DS0 + 3 * DS;      // dz_gate j=0,1                (DS)x2
constant constexpr int DE_ZN0  = DE_ZG0 + 2 * DS;      // dz_cand j=0,1                (DS)x2
constant constexpr int DE_QP   = DE_ZN0 + 2 * DS;      // dq_pred                      (D)
constant constexpr int DE_QS0  = DE_QP + D;            // dq_sig j=0..2                (D)x3
constant constexpr int DE_DG0  = DE_QS0 + 3 * D;       // d dG_j                       (DS)x3
constant constexpr int DE_DR0  = DE_DG0 + 3 * DS;      // d dR_j                       (DS)x3
constant constexpr int DE_MUG  = DE_DR0 + 3 * DS;      // d muG                        (DS)
constant constexpr int DE_MUR  = DE_MUG + DS;          // d muR                        (DS)
constant constexpr int DE_PPI  = DE_MUR + DS;          // d pre_pi                     (DS)
constant constexpr int DE_ZPI  = DE_PPI + DS;          // dz_pi (scalar)               (1)
constant constexpr int DE_H0   = DE_ZPI + 1;           // dh j=0,1                     (64)x2
constant constexpr int DE_ZH   = DE_H0 + 2 * 64;       // dz_h j=0,1 (scalars)         (2)
constant constexpr int DE      = DE_ZH + 2;            // 5443

// (W^T x)[i] += sum_o W[o, i] * x[o]  — transposed matvec (column reads; SLC-cached weights).
inline void matvec_t_add(threadgroup float *dst, device const float *wp,
                         threadgroup const float *src, int OUT, int IN, uint tid) {
    for (int i = (int)tid; i < IN; i += TPB) {
        float acc = 0.0f;
        for (int o = 0; o < OUT; ++o) acc = fma(wp[(long)o * IN + i], src[o], acc);
        dst[i] += acc;
    }
}

// LayerNorm backward with saved (mean, rstd): dv = rstd*(dyw - mean(dyw) - xhat*mean(dyw*xhat)),
// dyw = dy .* w, xhat = (v - mean)*rstd. v (the pre-LN input) is recomputed by the caller into
// `vin`; dy in `dy`; result ADDED into `dv`.
inline void layer_norm_bwd(threadgroup const float *dy, threadgroup const float *vin,
                           device const float *w, float mean, float rstd, int N,
                           threadgroup float *dv, threadgroup float *scratch,
                           threadgroup float *scal2, uint tid, uint lane, uint sid) {
    float p1 = 0.0f, p2 = 0.0f;
    for (int i = (int)tid; i < N; i += TPB) {
        const float dyw = dy[i] * w[i];
        const float xh = (vin[i] - mean) * rstd;
        p1 += dyw;
        p2 += dyw * xh;
    }
    const float m1 = block_sum(p1, scratch, tid, lane, sid) / N;
    const float m2 = block_sum(p2, scratch, tid, lane, sid) / N;
    for (int i = (int)tid; i < N; i += TPB) {
        const float dyw = dy[i] * w[i];
        const float xh = (vin[i] - mean) * rstd;
        dv[i] += rstd * (dyw - m1 - xh * m2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (void)scal2;
}

// dqr[h, n] = sum_p S[h, p, n] * dr[h, p]   (the transposed read)
inline void state_read_t(device const float *S, threadgroup const float *dr,
                         threadgroup float *dqr, uint tid) {
    for (int hn = (int)tid; hn < NH * DH; hn += TPB) {   // hn = h*DH + n
        const int h = hn / DH, n = hn % DH;
        float acc = 0.0f;
        for (int p = 0; p < DH; ++p) acc = fma(S[((long)h * DH + p) * DH + n], dr[h * DH + p], acc);
        dqr[hn] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// dq = R(-phi) dqr ; also emits nothing (host derives dcos/dsin by re-rotating emitted dq).
inline void unrotate(threadgroup const float *dqr, device const float *cosv,
                     device const float *sinv, threadgroup float *dq, uint tid) {
    for (int i = (int)tid; i < D / 2; i += TPB) {
        const float c = cosv[i], s = sinv[i];
        const float g0 = dqr[2 * i], g1 = dqr[2 * i + 1];
        dq[2 * i] = g0 * c + g1 * s;
        dq[2 * i + 1] = -g0 * s + g1 * c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void aum_silence_bwd(
    device const float *streams  [[buffer(0)]],   // (B, L, SW)
    device const float *alpha    [[buffer(1)]],
    device const float *xw       [[buffer(2)]],
    device const float *k_rot    [[buffer(3)]],
    device const float *wpack    [[buffer(4)]],
    device const float *save     [[buffer(5)]],   // (B, L, SV) forward save pack
    device const int   *j_star   [[buffer(6)]],
    device const float *S_ckpt   [[buffer(7)]],   // (B, nseg, NH, DH, DH)
    device const float *dout     [[buffer(8)]],   // (B, L, DO) incoming grads
    device       float *demit    [[buffer(9)]],   // (B, L, DE) emitted d-vectors
    device       float *dalpha   [[buffer(10)]],  // (B, L, NH)
    device       float *dxw      [[buffer(11)]],  // (B, L, NH*DH)
    device       float *dkrot    [[buffer(12)]],  // (B, L, NH*DH)
    device       float *S_seg    [[buffer(13)]],  // (B, 64, NH, DH, DH) segment replay scratch
    device       float *dS       [[buffer(14)]],  // (B, NH, DH, DH) chain workspace, zeroed
    constant     uint  &L        [[buffer(15)]],
    constant     float &kappa    [[buffer(16)]],
    constant     int   &forced   [[buffer(17)]],
    constant     int   &no_op    [[buffer(18)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sid  [[simdgroup_index_in_threadgroup]]) {

    const long b = tgid.z;
    const uint nseg = (L + 63) / 64;
    device float *dSb = dS + b * (long)(NH * DH * DH);
    device float *seg = S_seg + b * (long)(64 * NH * DH * DH);

    threadgroup float sig[(PJ + 1) * DS];        // saved sigma^0..2 of the token
    threadgroup float dsig[(PJ + 1) * DS];
    threadgroup float dcarry[DS];                // d sigma^{*}_t coming from token t+1
    threadgroup float dsprev[DS];                // d sigma_prev accumulated at this token
    threadgroup float sprev[DS];                 // sigma_prev (= sigma^*_{t-1})
    threadgroup float dr[(PJ + 1) * D];          // d r_j accumulators
    threadgroup float big_a[D], big_b[D], big_c[D];
    threadgroup float sm_a[DS], sm_b[DS], sm_c[DS];
    threadgroup float muG[DS], muR[DS];
    threadgroup float mu[DM], etil[DM], dmu[DM], detil[DM];
    threadgroup float hv[64], dhv[64];
    threadgroup float scratch[TPB / 32];
    threadgroup float scal[8];

    for (int i = (int)tid; i < DS; i += TPB) dcarry[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int s = (int)nseg - 1; s >= 0; --s) {
        // ---- replay S forward through this segment: seg[i] = S_{t0+i} (post-update) ----
        const int t0 = s * 64;
        const int tend = min((int)L, t0 + 64);
        device const float *ck = S_ckpt + (b * nseg + s) * (long)(NH * DH * DH);
        for (int i = t0; i < tend; ++i) {
            device const float *prev = (i == t0) ? ck : seg + (long)(i - 1 - t0) * NH * DH * DH;
            device float *cur = seg + (long)(i - t0) * NH * DH * DH;
            device const float *xt = xw + (b * L + i) * (long)(NH * DH);
            device const float *kt = k_rot + (b * L + i) * (long)(NH * DH);
            device const float *at = alpha + (b * L + i) * NH;
            for (int hp = (int)tid; hp < NH * DH; hp += TPB) {
                const int h = hp / DH;
                const float a = at[h], xv = xt[hp];
                device const float *pr = prev + (long)hp * DH;
                device float *cu = cur + (long)hp * DH;
                device const float *kh = kt + h * DH;
                for (int n = 0; n < DH; ++n) cu[n] = a * pr[n] + xv * kh[n];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (int t = tend - 1; t >= t0; --t) {
            device const float *st = streams + (b * L + t) * (long)SW;
            device const float *sv = save + (b * L + t) * (long)SV;
            device const float *dv = dout + (b * L + t) * (long)DO;
            device float *de = demit + (b * L + t) * (long)DE;
            device const float *S_t = seg + (long)(t - t0) * NH * DH * DH;
            device const float *S_p = (t == t0) ? ck : seg + (long)(t - 1 - t0) * NH * DH * DH;
            device const float *xt = xw + (b * L + t) * (long)(NH * DH);
            device const float *kt = k_rot + (b * L + t) * (long)(NH * DH);
            device const float *at = alpha + (b * L + t) * NH;
            const int js = no_op ? 0 : j_star[b * L + t];

            // load saved sigma trajectory, mu, e_tilde, sigma_prev; init accumulators
            for (int i = (int)tid; i < (PJ + 1) * DS; i += TPB) {
                sig[i] = sv[SV_SIGST + i];
                dsig[i] = dv[DO_SIG + i];
            }
            for (int i = (int)tid; i < DS; i += TPB) {
                sprev[i] = (t > 0) ? save[(b * L + t - 1) * (long)SV + SV_SSTAR + i] : 0.0f;
                dsprev[i] = 0.0f;
            }
            for (int i = (int)tid; i < DM; i += TPB) {
                mu[i] = sv[SV_MU + i];
                etil[i] = sv[SV_ETIL + i];
                dmu[i] = 0.0f;
                detil[i] = 0.0f;
            }
            for (int i = (int)tid; i < (PJ + 1) * D; i += TPB) dr[i] = dv[DO_RST + i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = (int)tid; i < DS; i += TPB)
                dsig[js * DS + i] += dcarry[i] + dv[DO_SST + i];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const float pi_v = sv[SV_PI];
            float dpi = dv[DO_PI];
            float dE_tot[PJ + 1];
            for (int j = 0; j <= PJ; ++j) dE_tot[j] = dv[DO_E + j];

            // ---- halting backward (skipped when forced: w was one-hot) ----
            if (forced < 0) {
                // recompute p0, p1 (and h_j) from the halt MLP
                float pj[2];
                for (int j = 0; j < PJ; ++j) {
                    for (int i = (int)tid; i < 64; i += TPB) hv[i] = 0.0f;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    matvec_add(hv, wpack + W_H1S, sig + j * DS, 64, DS, tid);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    float part = 0.0f;
                    for (int i = (int)tid; i < 64; i += TPB) {
                        hv[i] = tanh(hv[i] + wpack[W_H1PI + i] * pi_v
                                     + wpack[W_H1E + i] * sv[SV_E + j]);
                        part += wpack[W_H2 + i] * hv[i];
                    }
                    const float z = block_sum(part, scratch, tid, lane, sid);
                    pj[j] = 1.0f / (1.0f + exp(-z));
                    // stash h_j for the gradient pass below (hv reused per j — process grads now)
                    const float dw0 = dv[DO_W + 0], dw1 = dv[DO_W + 1], dw2 = dv[DO_W + 2];
                    float dp;
                    if (j == 0) {
                        // needs p1 too — defer j=0's dp to after the j loop? Instead compute
                        // both p first. Handled by the two-pass structure below.
                        dp = 0.0f; (void)dw0; (void)dw1; (void)dw2;
                    }
                    (void)dp;
                    if (j == 0) { for (int i = (int)tid; i < 64; i += TPB) dhv[i] = hv[i]; }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                // dhv currently holds h_0; hv holds h_1. Cascade grads:
                const float p0 = pj[0], p1 = pj[1];
                const float dw0 = dv[DO_W + 0], dw1 = dv[DO_W + 1], dw2 = dv[DO_W + 2];
                const float dp0 = dw0 - dw1 * p1 - dw2 * (1.0f - p1);
                const float dp1 = dw1 * (1.0f - p0) - dw2 * (1.0f - p0);
                const float dz0 = p0 * (1.0f - p0) * dp0;
                const float dz1 = p1 * (1.0f - p1) * dp1;
                if (tid == 0) { de[DE_ZH + 0] = dz0; de[DE_ZH + 1] = dz1; }
                // per-j: dh = (1 - h^2) * w_h2 * dz ; dsig_j += Wh1s^T dh ; dpi/dE accumulate
                for (int j = 0; j < PJ; ++j) {
                    threadgroup float *hj = (j == 0) ? dhv : hv;
                    const float dz = (j == 0) ? dz0 : dz1;
                    float ppi = 0.0f, pE = 0.0f;
                    for (int i = (int)tid; i < 64; i += TPB) {
                        const float dh = (1.0f - hj[i] * hj[i]) * wpack[W_H2 + i] * dz;
                        hj[i] = dh;                                   // reuse as dh
                        de[DE_H0 + j * 64 + i] = dh;
                        ppi += wpack[W_H1PI + i] * dh;
                        pE += wpack[W_H1E + i] * dh;
                    }
                    dpi += block_sum(ppi, scratch, tid, lane, sid);
                    dE_tot[j] += block_sum(pE, scratch, tid, lane, sid);
                    matvec_t_add(dsig + j * DS, wpack + W_H1S, hj, 64, DS, tid);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            } else {
                if (tid == 0) { de[DE_ZH] = 0.0f; de[DE_ZH + 1] = 0.0f; }
                for (int i = (int)tid; i < 2 * 64; i += TPB) de[DE_H0 + i] = 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // ---- pressure backward: needs de_n = |e~|, dsR = |P_R r_0 - Q_R sigma^0| ----
            {
                float p2 = 0.0f;
                for (int i = (int)tid; i < DM; i += TPB) p2 += etil[i] * etil[i];
                const float de_n = sqrt(block_sum(p2, scratch, tid, lane, sid));
                // dR_0 into sm_a
                for (int i = (int)tid; i < DS; i += TPB) sm_a[i] = 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                matvec_add(sm_a, wpack + W_QR, sig, DS, DS, tid);   // Q_R sigma^0
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float p3 = 0.0f;
                for (int i = (int)tid; i < DS; i += TPB) {
                    float pr = 0.0f;
                    device const float *row = wpack + W_PRR + (long)i * D;
                    device const float *r0 = sv + SV_RST;
                    for (int n = 0; n < D; ++n) pr = fma(row[n], r0[n], pr);
                    sm_a[i] = pr - sm_a[i];                          // dR_0
                    p3 += sm_a[i] * sm_a[i];
                }
                const float dsR = sqrt(block_sum(p3, scratch, tid, lane, sid));
                const float dz_pi = (1.0f - exp(-pi_v)) * dpi;       // softplus' = sigmoid(z)
                if (tid == 0) de[DE_ZPI] = dz_pi;
                // dpre_pi_i = w_pi_i * (1 - tanh^2(pre_i)) * dz ; collect d_de, d_dsR
                float pde = 0.0f, pds = 0.0f;
                for (int i = (int)tid; i < DS; i += TPB) {
                    const float pre = st[SW_CPRESS + i] + wpack[W_WPDE + i] * de_n
                                      + wpack[W_WPDS + i] * dsR;
                    const float th = tanh(pre);
                    const float dpre = wpack[W_WPI + i] * (1.0f - th * th) * dz_pi;
                    de[DE_PPI + i] = dpre;
                    pde += wpack[W_WPDE + i] * dpre;
                    pds += wpack[W_WPDS + i] * dpre;
                }
                const float d_de = block_sum(pde, scratch, tid, lane, sid);
                const float d_ds = block_sum(pds, scratch, tid, lane, sid);
                // d e~ += (e~/|e~|) d_de ; fold (dR_0/|dR_0|) d_dsR into the E-path ddR_0 below
                if (de_n > 0.0f)
                    for (int i = (int)tid; i < DM; i += TPB) detil[i] += etil[i] / de_n * d_de;
                if (tid == 0) scal[0] = (dsR > 0.0f) ? d_ds / dsR : 0.0f;   // ddR0 scale
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // keep dR_0 (sm_a) for the consistency pass: stash into big_c[0..DS)
                for (int i = (int)tid; i < DS; i += TPB) big_c[i] = sm_a[i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // ---- consistency backward (mu detached) ----
            for (int i = (int)tid; i < DS; i += TPB) { muG[i] = 0.0f; muR[i] = 0.0f; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            matvec_add(muG, wpack + W_PCG, mu, DS, DM, tid);
            matvec_add(muR, wpack + W_PCR, mu, DS, DM, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = (int)tid; i < DS; i += TPB) { sm_b[i] = 0.0f; sm_c[i] = 0.0f; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int j = 0; j <= PJ; ++j) {
                threadgroup float *sj = sig + j * DS;
                // dG_j = c_pg - Q_G sigma^j (sm_a); dR_j (reuse big_c for j=0, else recompute)
                for (int i = (int)tid; i < DS; i += TPB) sm_a[i] = 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                matvec_add(sm_a, wpack + W_QG, sj, DS, DS, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int i = (int)tid; i < DS; i += TPB) sm_a[i] = st[SW_CPG + i] - sm_a[i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                threadgroup float *dRj = big_c;                       // (DS floats reused)
                if (j > 0) {
                    for (int i = (int)tid; i < DS; i += TPB) big_c[i] = 0.0f;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    matvec_add(big_c, wpack + W_QR, sj, DS, DS, tid);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (int i = (int)tid; i < DS; i += TPB) {
                        float pr = 0.0f;
                        device const float *row = wpack + W_PRR + (long)i * D;
                        device const float *rj = sv + SV_RST + j * D;
                        for (int n = 0; n < D; ++n) pr = fma(row[n], rj[n], pr);
                        big_c[i] = pr - big_c[i];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                // ddG (into sm_a, overwriting dG), ddR (dRj -> overwritten), dmu accumulators
                for (int i = (int)tid; i < DS; i += TPB) {
                    const float dG = sm_a[i], dRv = dRj[i];
                    const float ddG = 2.0f * muG[i] * muG[i] * dG * dE_tot[j];
                    float ddR = 2.0f * muR[i] * muR[i] * dRv * dE_tot[j];
                    if (j == 0) ddR += scal[0] * dRv;                 // the dsR fold
                    sm_b[i] += 2.0f * muG[i] * dG * dG * dE_tot[j];   // dmuG
                    sm_c[i] += 2.0f * muR[i] * dRv * dRv * dE_tot[j]; // dmuR
                    de[DE_DG0 + j * DS + i] = ddG;
                    de[DE_DR0 + j * DS + i] = ddR;
                    sm_a[i] = ddG;
                    dRj[i] = ddR;
                    // kappa term
                    const float dk = 2.0f * kappa * (sj[i] - sprev[i]) * dE_tot[j];
                    dsig[j * DS + i] += dk;
                    dsprev[i] -= dk;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // dsig_j -= Q_G^T ddG + Q_R^T ddR ; dr_j += P_R^T ddR
                for (int i = (int)tid; i < DS; i += TPB) {
                    float a1 = 0.0f, a2 = 0.0f;
                    for (int o = 0; o < DS; ++o) {
                        a1 = fma(wpack[W_QG + (long)o * DS + i], sm_a[o], a1);
                        a2 = fma(wpack[W_QR + (long)o * DS + i], dRj[o], a2);
                    }
                    dsig[j * DS + i] -= a1 + a2;
                }
                matvec_t_add(dr + j * D, wpack + W_PRR, dRj, DS, D, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (int i = (int)tid; i < DS; i += TPB) {
                de[DE_MUG + i] = sm_b[i];
                de[DE_MUR + i] = sm_c[i];
            }

            // ---- interleaved reverse through the revision loop ----
            // dsig[j] is complete only after BOTH its E/halting/output grads AND the read-bwd of
            // r_j (dq_j -> W_qsig^T) have landed; dr[j] is complete only after the revise-bwd
            // that consumed r_j. The correct order is therefore:
            //   read(2) -> revise(1) -> read(1) -> revise(0) -> read(0) -> init
            for (int j = PJ; j >= 0; --j) {
                // (a) read backward for r_j (dr[j] is final here)
                for (int i = (int)tid; i < D; i += TPB) big_a[i] = 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                matvec_add(big_a, wpack + W_QSIG, sig + j * DS, D, DS, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                rotate_query(big_a, st + SW_COST, st + SW_SINT, big_b, tid);   // qr_j
                for (int hp = (int)tid; hp < NH * DH; hp += TPB) {             // dS_t += dr (x) qr
                    const int h = hp / DH;
                    const float drv = dr[j * D + hp];
                    device float *dSrow = dSb + (long)hp * DH;
                    for (int n = 0; n < DH; ++n) dSrow[n] += drv * big_b[h * DH + n];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                state_read_t(S_t, dr + j * D, big_a, tid);                     // dqr_j
                unrotate(big_a, st + SW_COST, st + SW_SINT, big_b, tid);       // dq_j
                for (int i = (int)tid; i < D; i += TPB) de[DE_QS0 + j * D + i] = big_b[i];
                matvec_t_add(dsig + j * DS, wpack + W_QSIG, big_b, D, DS, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (j == 0) break;

                // (b) revision backward for step jr = j-1 (consumes the now-complete dsig[j])
                {
                    const int jr = j - 1;
                    threadgroup float *sj = sig + jr * DS;
                    device const float *rj = sv + SV_RST + jr * D;
                    // recompute gate (sm_a) / cand (sm_b) pre-activations
                    for (int i = (int)tid; i < DS; i += TPB) {
                        sm_a[i] = st[SW_CGATE + i];
                        sm_b[i] = st[SW_CCAND + i];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    matvec_add(sm_a, wpack + W_GS, sj, DS, DS, tid);
                    matvec_add(sm_a, wpack + W_GET, etil, DS, DM, tid);
                    matvec_add(sm_a, wpack + W_GMU, mu, DS, DM, tid);
                    matvec_add(sm_b, wpack + W_NS, sj, DS, DS, tid);
                    matvec_add(sm_b, wpack + W_NET, etil, DS, DM, tid);
                    matvec_add(sm_b, wpack + W_NMU, mu, DS, DM, tid);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (int i = (int)tid; i < DS; i += TPB) {
                        float ag = 0.0f, an = 0.0f;
                        device const float *rowg = wpack + W_GR + (long)i * D;
                        device const float *rown = wpack + W_NR + (long)i * D;
                        for (int n = 0; n < D; ++n) {
                            ag = fma(rowg[n], rj[n], ag);
                            an = fma(rown[n], rj[n], an);
                        }
                        const float gate = 1.0f / (1.0f + exp(-(sm_a[i] + ag)));
                        const float cand = tanh(sm_b[i] + an);
                        sm_a[i] = gate;
                        sm_b[i] = cand;
                        sm_c[i] = sj[i] + gate * cand;                // pre-LN input v
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    // LN1 backward: dy = dsig[jr+1]; emit dy; dv accumulates into dsig[jr]
                    for (int i = (int)tid; i < DS; i += TPB) de[DE_DS0 + (jr + 1) * DS + i]
                        = dsig[(jr + 1) * DS + i];
                    for (int i = (int)tid; i < DS; i += TPB) big_c[i] = 0.0f;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    layer_norm_bwd(dsig + (jr + 1) * DS, sm_c, wpack + W_LN1W,
                                   sv[SV_LN1 + 2 * (jr + 1)], sv[SV_LN1 + 2 * (jr + 1) + 1], DS,
                                   big_c, scratch, scal, tid, lane, sid);
                    for (int i = (int)tid; i < DS; i += TPB) {
                        const float dvv = big_c[i];
                        const float gate = sm_a[i], cand = sm_b[i];
                        dsig[jr * DS + i] += dvv;                     // through the residual
                        const float dgate = dvv * cand, dcand = dvv * gate;
                        sm_a[i] = gate * (1.0f - gate) * dgate;       // dz_g
                        sm_b[i] = (1.0f - cand * cand) * dcand;       // dz_n
                        de[DE_ZG0 + jr * DS + i] = sm_a[i];
                        de[DE_ZN0 + jr * DS + i] = sm_b[i];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    matvec_t_add(dsig + jr * DS, wpack + W_GS, sm_a, DS, DS, tid);
                    matvec_t_add(dsig + jr * DS, wpack + W_NS, sm_b, DS, DS, tid);
                    matvec_t_add(detil, wpack + W_GET, sm_a, DS, DM, tid);
                    matvec_t_add(detil, wpack + W_NET, sm_b, DS, DM, tid);
                    matvec_t_add(dmu, wpack + W_GMU, sm_a, DS, DM, tid);
                    matvec_t_add(dmu, wpack + W_NMU, sm_b, DS, DM, tid);
                    matvec_t_add(dr + jr * D, wpack + W_GR, sm_a, DS, D, tid);
                    matvec_t_add(dr + jr * D, wpack + W_NR, sm_b, DS, D, tid);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }

            // ---- sigma^0 (init) backward ----
            for (int i = (int)tid; i < DS; i += TPB) de[DE_DS0 + i] = dsig[i];
            // recompute z_init = c_init + Wi_s sprev + Wi_et e~ + Wi_mu mu
            for (int i = (int)tid; i < DS; i += TPB) sm_c[i] = st[SW_CINIT + i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            matvec_add(sm_c, wpack + W_IS, sprev, DS, DS, tid);
            matvec_add(sm_c, wpack + W_IET, etil, DS, DM, tid);
            matvec_add(sm_c, wpack + W_IMU, mu, DS, DM, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = (int)tid; i < DS; i += TPB) sm_a[i] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            layer_norm_bwd(dsig, sm_c, wpack + W_LN1W, sv[SV_LN1], sv[SV_LN1 + 1], DS,
                           sm_a, scratch, scal, tid, lane, sid);                 // dz_init
            for (int i = (int)tid; i < DS; i += TPB) de[DE_ZI + i] = sm_a[i];
            matvec_t_add(dsprev, wpack + W_IS, sm_a, DS, DS, tid);
            matvec_t_add(detil, wpack + W_IET, sm_a, DS, DM, tid);
            matvec_t_add(dmu, wpack + W_IMU, sm_a, DS, DM, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- e_tilde / mu backward ----
            // e~ = mu (.) (W_err e); e = g - g_hat (both saved/streamed)
            for (int i = (int)tid; i < D; i += TPB) big_a[i] = st[SW_G + i] - sv[SV_GHAT + i]; // e
            for (int i = (int)tid; i < DM; i += TPB) sm_a[i] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            matvec_add(sm_a, wpack + W_ERR, big_a, DM, D, tid);       // W_err e
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = (int)tid; i < DM; i += TPB) {
                dmu[i] += sm_a[i] * detil[i];
                sm_b[i] = mu[i] * detil[i];                           // d(W_err e)
                de[DE_ETT + i] = detil[i];
                sm_c[i] = mu[i] * (1.0f - mu[i]) * dmu[i];            // dz_mu
                de[DE_ZMU + i] = sm_c[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // de_vec = W_err^T (mu.detil) + W_mue^T dz_mu   (into big_b)
            for (int i = (int)tid; i < D; i += TPB) big_b[i] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            matvec_t_add(big_b, wpack + W_ERR, sm_b, DM, D, tid);
            matvec_t_add(big_b, wpack + W_MUE, sm_c, DM, D, tid);
            matvec_t_add(dsprev, wpack + W_MUS, sm_c, DM, DS, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- g_hat backward ----
            // dGhat_tot = dghat_in - de_vec
            for (int i = (int)tid; i < D; i += TPB) {
                big_a[i] = dv[DO_GHAT + i] - big_b[i];
                de[DE_GHT + i] = big_a[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // dy5 = W_P^T dGhat_tot
            for (int i = (int)tid; i < D; i += TPB) big_c[i] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            matvec_t_add(big_c, wpack + W_P, big_a, D, D, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // recompute pre = W_R r_pred + W_hyp sprev + c_phase (into big_a)
            for (int i = (int)tid; i < D; i += TPB) big_a[i] = st[SW_CPHASE + i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                // W_R r_pred: r_pred saved (device); row-major matvec with device src
                for (int o = (int)tid; o < D; o += TPB) {
                    float acc = 0.0f;
                    device const float *row = wpack + W_R + (long)o * D;
                    device const float *rp = sv + SV_RPRED;
                    for (int i2 = 0; i2 < D; ++i2) acc = fma(row[i2], rp[i2], acc);
                    big_a[o] += acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                matvec_add(big_a, wpack + W_HYP, sprev, D, DS, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (int i = (int)tid; i < D; i += TPB) big_b[i] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            layer_norm_bwd(big_c, big_a, wpack + W_LN5W, sv[SV_LN5], sv[SV_LN5 + 1], D,
                           big_b, scratch, scal, tid, lane, sid);                // dpre
            for (int i = (int)tid; i < D; i += TPB) de[DE_PRE + i] = big_b[i];
            matvec_t_add(dsprev, wpack + W_HYP, big_b, D, DS, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // dr_pred = W_R^T dpre (into big_a)
            for (int i = (int)tid; i < D; i += TPB) big_a[i] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            matvec_t_add(big_a, wpack + W_R, big_b, D, D, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- S-chain step at token t ----
            // dalpha/dxw/dkrot from dS_t and S_{t-1}
            for (int h = 0; h < NH; ++h) {
                float part = 0.0f;
                for (int pn = (int)tid; pn < DH * DH; pn += TPB)
                    part += dSb[(long)h * DH * DH + pn] * S_p[(long)h * DH * DH + pn];
                const float da = block_sum(part, scratch, tid, lane, sid);
                if (tid == 0) dalpha[(b * L + t) * NH + h] = da;
            }
            for (int hp = (int)tid; hp < NH * DH; hp += TPB) {
                const int h = hp / DH;
                device const float *dSrow = dSb + (long)hp * DH;
                device const float *kh = kt + h * DH;
                float acc = 0.0f;
                for (int n = 0; n < DH; ++n) acc = fma(dSrow[n], kh[n], acc);
                dxw[(b * L + t) * (long)(NH * DH) + hp] = acc;
            }
            for (int hn = (int)tid; hn < NH * DH; hn += TPB) {
                const int h = hn / DH, n = hn % DH;
                float acc = 0.0f;
                for (int p = 0; p < DH; ++p)
                    acc = fma(dSb[((long)h * DH + p) * DH + n], xt[h * DH + p], acc);
                dkrot[(b * L + t) * (long)(NH * DH) + hn] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // dS <- alpha_t (.) dS  (now dS_{t-1} sans the predictive read)
            for (int hp = (int)tid; hp < NH * DH; hp += TPB) {
                const int h = hp / DH;
                const float a = at[h];
                device float *dSrow = dSb + (long)hp * DH;
                for (int n = 0; n < DH; ++n) dSrow[n] *= a;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- predictive-read backward (reads S_{t-1}; contributes to dS_{t-1}) ----
            // q_pred = W_qpred sprev ; qr = R(phi_{t-1}) q
            for (int i = (int)tid; i < D; i += TPB) big_c[i] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            matvec_add(big_c, wpack + W_QPRED, sprev, D, DS, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            rotate_query(big_c, st + SW_COSP, st + SW_SINP, big_b, tid);        // qr_pred
            for (int hp = (int)tid; hp < NH * DH; hp += TPB) {
                const int h = hp / DH;
                const float drv = big_a[hp];                                    // dr_pred
                device float *dSrow = dSb + (long)hp * DH;
                for (int n = 0; n < DH; ++n) dSrow[n] += drv * big_b[h * DH + n];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            state_read_t(S_p, big_a, big_c, tid);                               // dqr_pred
            unrotate(big_c, st + SW_COSP, st + SW_SINP, big_b, tid);            // dq_pred
            for (int i = (int)tid; i < D; i += TPB) de[DE_QP + i] = big_b[i];
            matvec_t_add(dsprev, wpack + W_QPRED, big_b, D, DS, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- carry to t-1 ----
            for (int i = (int)tid; i < DS; i += TPB) dcarry[i] = dsprev[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
