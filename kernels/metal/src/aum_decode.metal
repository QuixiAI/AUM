#include <metal_stdlib>
using namespace metal;

// AUM-Ø single-token U-phase decode step (§6), plain per-(batch,head) kernel.
//
// The evidence state S[p,n] is (Dv x Dqk); with the AUM head geometry Dv == Dqk == D. Given the
// current token's affine decay alpha = exp(-lambda*tau), the write vector x = rho*tau*v_hat (per p),
// the rotated key k_rot (per n) and rotated query q_rot (per n), one step is:
//
//     S[p,n] <- alpha * S[p,n] + x[p] * k_rot[n]        (affine decay + rank-1 write)
//     out[p]  = sum_n S[p,n] * q_rot[n]                 (readout S_t . R(phi) q, AFTER the write)
//
// One threadgroup per (batch, head); one thread per output row p (D threads). Each thread owns the
// full row S[p, :], so the decay, the rank-1 update and the matvec readout are all independent
// across rows — no cross-thread reduction. q_rot / k_rot (shared over all rows) are staged into
// threadgroup memory once. State is fp32 (a recurrence read/written every step). The dynamics,
// rotary, k/v L2-norm, the D-skip (out += D_h * v) and the gated-RMSNorm are done by the caller.
template <int D>
kernel void aum_decode(device       float *S      [[buffer(0)]],   // (B,H,D,D) in/out, S[p,n]
                       device const float *alpha  [[buffer(1)]],   // (B,H)     decay = exp(-lambda*tau)
                       device const float *x      [[buffer(2)]],   // (B,H,D)   rho*tau*v_hat, per p
                       device const float *k_rot  [[buffer(3)]],   // (B,H,D)   per n
                       device const float *q_rot  [[buffer(4)]],   // (B,H,D)   per n
                       device       float *out    [[buffer(5)]],   // (B,H,D)   per p
                       constant unsigned  &H      [[buffer(6)]],
                       uint3 blockIdx [[threadgroup_position_in_grid]],
                       uint  tid      [[thread_index_in_threadgroup]]) {
    const uint h  = blockIdx.y;
    const uint b  = blockIdx.z;
    const uint bh = b * H + h;

    threadgroup float kk[D];        // k_rot, shared across all rows p
    threadgroup float qq[D];        // q_rot
    kk[tid] = k_rot[bh * D + tid];
    qq[tid] = q_rot[bh * D + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint  p    = tid;                       // this thread owns output row p in [0, D)
    const float a    = alpha[bh];
    const float xp   = x[bh * D + p];
    device float* Srow = S + (bh * D + p) * D;    // S[p, :]
    float acc = 0.0f;
    for (uint n = 0; n < D; ++n) {
        float s = a * Srow[n] + xp * kk[n];       // decayed state + rank-1 write
        Srow[n] = s;                              // persist the new state
        acc += s * qq[n];                         // readout against q_rot
    }
    out[bh * D + p] = acc;
}

#define instantiate_aum_decode(D)                                        \
  template [[host_name("aum_decode_" #D)]] [[kernel]] void               \
  aum_decode<D>(device float *S [[buffer(0)]], device const float *alpha [[buffer(1)]], \
    device const float *x [[buffer(2)]], device const float *k_rot [[buffer(3)]], \
    device const float *q_rot [[buffer(4)]], device float *out [[buffer(5)]], \
    constant unsigned &H [[buffer(6)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]);

instantiate_aum_decode(64);
instantiate_aum_decode(128);
