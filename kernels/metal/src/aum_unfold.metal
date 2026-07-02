#include <metal_stdlib>
using namespace metal;

// AUM-Ø U-phase fused operand pipeline (§4). Two plain kernels (one simdgroup per (b, l, h) row;
// elementwise + simd row reductions — no tile machinery needed) that eliminate every host pass
// around the SSD core:
//
//   aum_operands:  C = R(phi) q,  B = R(phi) (k/||k||),  X = rho*tau * (v/||v||)
//                  reading the model's (B, N, H*D) layout and writing the kernel's (B, H, N, D)
//                  layout — the transpose is free, killing the host rearranges too.
//   aum_epilogue:  out = silu(z) * RMSNorm(Y + Dskip*v) * w_norm
//                  reading Y (B, H, N, D) and v/z (B, N, H*D), writing (B, N, H*D).
//
// The rotation ladder is applied per adjacent pair (2m, 2m+1) with angle phi * freqs[m]; each
// lane owns a contiguous span of D/32 elements, so pairs never split across lanes (D/32 is even
// for D in {64, 128}). Norm reductions via simd_sum. All math fp32, I/O bf16 (phi/freqs fp32).

kernel void aum_operands(device const bfloat *q     [[buffer(0)]],   // (B, N, H*D)
                         device const bfloat *k     [[buffer(1)]],
                         device const bfloat *v     [[buffer(2)]],
                         device const float  *phi   [[buffer(3)]],   // (B, N, H)
                         device const float  *rtw   [[buffer(4)]],   // rho*tau (B, N, H)
                         device const float  *freqs [[buffer(5)]],   // (D/2) ladder
                         device bfloat       *C     [[buffer(6)]],   // (B, H, N, D)
                         device bfloat       *Bo    [[buffer(7)]],
                         device bfloat       *X     [[buffer(8)]],
                         constant unsigned   &N     [[buffer(9)]],
                         constant unsigned   &H     [[buffer(10)]],
                         constant unsigned   &D     [[buffer(11)]],
                         constant float      &eps   [[buffer(12)]],
                         uint3 gid  [[threadgroup_position_in_grid]],
                         uint  lane [[thread_index_in_simdgroup]]) {
    const long b = gid.z, h = gid.y, l = gid.x;
    const long in_base  = ((b * N + l) * H + h) * D;
    const long out_base = ((b * H + h) * N + l) * D;
    const int  per = (int)D / 32;                      // elements per lane (2 or 4, even)

    // row norms of k and v
    float kss = 0.0f, vss = 0.0f;
    for (int i = 0; i < per; ++i) {
        const int c = (int)lane * per + i;
        const float kv_ = float(k[in_base + c]);
        const float vv_ = float(v[in_base + c]);
        kss += kv_ * kv_;
        vss += vv_ * vv_;
    }
    kss = simd_sum(kss);
    vss = simd_sum(vss);
    const float kinv = 1.0f / (metal::sqrt(kss) + eps);
    const float vinv = 1.0f / (metal::sqrt(vss) + eps);

    const float ph = phi[(b * N + l) * H + h];
    const float w  = rtw[(b * N + l) * H + h];

    for (int i = 0; i < per; i += 2) {                 // one rotation pair per iteration
        const int c = (int)lane * per + i;
        const float ang = ph * freqs[c >> 1];
        const float cs = metal::cos(ang), sn = metal::sin(ang);
        const float q0 = float(q[in_base + c]), q1 = float(q[in_base + c + 1]);
        const float k0 = float(k[in_base + c]) * kinv, k1 = float(k[in_base + c + 1]) * kinv;
        C[out_base + c]     = bfloat(q0 * cs - q1 * sn);
        C[out_base + c + 1] = bfloat(q0 * sn + q1 * cs);
        Bo[out_base + c]     = bfloat(k0 * cs - k1 * sn);
        Bo[out_base + c + 1] = bfloat(k0 * sn + k1 * cs);
        X[out_base + c]     = bfloat(w * float(v[in_base + c]) * vinv);
        X[out_base + c + 1] = bfloat(w * float(v[in_base + c + 1]) * vinv);
    }
}

kernel void aum_epilogue(device const bfloat *Y     [[buffer(0)]],   // (B, H, N, D)
                         device const bfloat *v     [[buffer(1)]],   // (B, N, H*D)
                         device const bfloat *z     [[buffer(2)]],   // (B, N, H*D)
                         device const float  *Dsk   [[buffer(3)]],   // D-skip, flat (H*D)
                         device const float  *wn    [[buffer(4)]],   // gated-RMSNorm weight (D)
                         device bfloat       *out   [[buffer(5)]],   // (B, N, H*D)
                         constant unsigned   &N     [[buffer(6)]],
                         constant unsigned   &H     [[buffer(7)]],
                         constant unsigned   &D     [[buffer(8)]],
                         constant float      &eps   [[buffer(9)]],   // RMSNorm eps (1e-5)
                         uint3 gid  [[threadgroup_position_in_grid]],
                         uint  lane [[thread_index_in_simdgroup]]) {
    const long b = gid.z, h = gid.y, l = gid.x;
    const long y_base = ((b * H + h) * N + l) * D;
    const long r_base = ((b * N + l) * H + h) * D;
    const int  per = (int)D / 32;

    // y' = Y + Dskip * v ; rms over the row
    float ss = 0.0f;
    float yv[4];                                       // per <= 4
    for (int i = 0; i < per; ++i) {
        const int c = (int)lane * per + i;
        const float yy = float(Y[y_base + c]) + Dsk[h * D + c] * float(v[r_base + c]);
        yv[i] = yy;
        ss += yy * yy;
    }
    ss = simd_sum(ss);
    const float inv = metal::rsqrt(ss / (float)D + eps);
    for (int i = 0; i < per; ++i) {
        const int c = (int)lane * per + i;
        const float zz = float(z[r_base + c]);
        const float gate = zz / (1.0f + metal::exp(-zz));          // silu(z)
        out[r_base + c] = bfloat(yv[i] * inv * wn[c] * gate);
    }
}
