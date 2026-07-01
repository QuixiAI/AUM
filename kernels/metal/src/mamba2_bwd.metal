#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// Mamba-2 / SSD BACKWARD (dC, dB, dX), bf16, D in {64,128}. Companion to mamba2.metal.
// Forward: P = C@Bᵀ, L[i,j] = exp(cl_i - cl_j) (causal), M = P⊙L, Y = M@X.
// Given dY:  dM = (dY@Xᵀ)⊙causal ; dP = dM⊙L ; dC = dP@B ; dB = dPᵀ@C ; dX = Mᵀ@dY.
// SSD is linear (no softmax) so there is no logsumexp/delta pass. dcumlog is computed on the host
// via the identity dcl[k] = <dY_k,Y_k> - <dX_k,X_k> (rowsum/colsum of dM⊙M), so the kernels emit
// only the D-dimensional grads (standard (B,H,N,D) stores) — no row/col-vec reductions here.
// Two atomics-free kernels (one simdgroup per 8-row block), mirroring attn_bwd dq/dkv:
//   mamba2_bwd_i — fix query-chunk i, loop j<=i, accumulate dC_i.
//   mamba2_bwd_j — fix key-chunk j, loop i>=j, accumulate dB_j and dX_j.

constant constexpr const int TB = 8;

// ---------- dC: fix query block i, loop key j<=i ----------
template <int D>
kernel void mamba2_bwd_i(device   bf16     *C  [[buffer(0)]],
                         device   bf16     *Bm [[buffer(1)]],
                         device   bf16     *X  [[buffer(2)]],
                         device   bf16     *dY [[buffer(3)]],
                         device   float    *cl [[buffer(4)]],
                         device   bf16     *dC [[buffer(5)]],
                         constant unsigned &N  [[buffer(6)]],
                         constant unsigned &H  [[buffer(7)]],
                         uint3 blockIdx [[threadgroup_position_in_grid]],
                         uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "mamba2_bwd: D must be 64 or 128");
    using gl_t  = gl<bfloat, 1, -1, -1, D>;
    using gl_cl = gl<float, 1, -1, 1, -1>;
    gl_t gC(C, nullptr, H, N, nullptr), gB(Bm, nullptr, H, N, nullptr), gX(X, nullptr, H, N, nullptr);
    gl_t gdY(dY, nullptr, H, N, nullptr), gdC(dC, nullptr, H, N, nullptr);
    gl_cl gcl(cl, nullptr, H, nullptr, N);

    const int b = blockIdx.z, h = blockIdx.y, i = blockIdx.x;
    rt_bf<TB, D> c_row, dy_row;
    load(c_row, gC, {b, h, i, 0}, laneId);
    load(dy_row, gdY, {b, h, i, 0}, laneId);
    typename rt_fl<TB, TB>::col_vec cl_i;
    load(cl_i, gcl, {b, h, 0, i}, laneId);

    rt_fl<TB, D> dc_reg;
    zero(dc_reg);
    for (int j = 0; j <= i; j++) {
        rt_bf<TB, D> b_row;
        load(b_row, gB, {b, h, j, 0}, laneId);
        rt_bf<TB, D, ducks::rt_layout::col> b_col, x_col;
        swap_layout(b_col, b_row, laneId);
        load(x_col, gX, {b, h, j, 0}, laneId);
        typename rt_fl<TB, TB>::row_vec cl_j;
        load(cl_j, gcl, {b, h, 0, j}, laneId);

        rt_fl<TB, TB> P, Ld, dM;
        zero(P);  mma_ABt(P, c_row, b_col, P);             // P = C_i·Bᵀ_j  (i×j)
        zero(Ld); add_row(Ld, Ld, cl_i); sub_col(Ld, Ld, cl_j); exp(Ld, Ld);  // L = exp(cl_i−cl_j)
        zero(dM); mma_ABt(dM, dy_row, x_col, dM);          // dM = dY_i·Xᵀ_j  (i×j)
        if (j == i) { float zf = 0.0f; make_causal(dM, dM, laneId, zf); }
        rt_fl<TB, TB> dP;
        mul(dP, dM, Ld);                                   // dP = dM⊙L
        rt_bf<TB, TB> dP_bf;
        copy(dP_bf, dP);
        mma_AB(dc_reg, dP_bf, b_row, dc_reg);              // dC_i += dP·B_j  (i×D)
    }
    store(gdC, dc_reg, {b, h, i, 0}, laneId);
}

// ---------- dB, dX: fix key block j, loop query i>=j ----------
template <int D>
kernel void mamba2_bwd_j(device   bf16     *C  [[buffer(0)]],
                         device   bf16     *Bm [[buffer(1)]],
                         device   bf16     *X  [[buffer(2)]],
                         device   bf16     *dY [[buffer(3)]],
                         device   float    *cl [[buffer(4)]],
                         device   bf16     *dB [[buffer(5)]],
                         device   bf16     *dX [[buffer(6)]],
                         constant unsigned &N  [[buffer(7)]],
                         constant unsigned &H  [[buffer(8)]],
                         uint3 blockIdx [[threadgroup_position_in_grid]],
                         uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "mamba2_bwd: D must be 64 or 128");
    using gl_t  = gl<bfloat, 1, -1, -1, D>;
    using gl_cl = gl<float, 1, -1, 1, -1>;
    gl_t gC(C, nullptr, H, N, nullptr), gB(Bm, nullptr, H, N, nullptr), gX(X, nullptr, H, N, nullptr);
    gl_t gdY(dY, nullptr, H, N, nullptr), gdB(dB, nullptr, H, N, nullptr), gdX(dX, nullptr, H, N, nullptr);
    gl_cl gcl(cl, nullptr, H, nullptr, N);

    const int b = blockIdx.z, h = blockIdx.y, j = blockIdx.x;
    rt_bf<TB, D> b_row;
    load(b_row, gB, {b, h, j, 0}, laneId);
    rt_bf<TB, D, ducks::rt_layout::col> b_col, x_col;
    swap_layout(b_col, b_row, laneId);
    load(x_col, gX, {b, h, j, 0}, laneId);
    typename rt_fl<TB, TB>::row_vec cl_j;
    load(cl_j, gcl, {b, h, 0, j}, laneId);

    rt_fl<TB, D> db_reg, dx_reg;
    zero(db_reg); zero(dx_reg);
    const int q_blocks = N / TB;
    for (int i = j; i < q_blocks; i++) {
        rt_bf<TB, D> c_row, dy_row;
        load(c_row, gC, {b, h, i, 0}, laneId);
        load(dy_row, gdY, {b, h, i, 0}, laneId);
        typename rt_fl<TB, TB>::col_vec cl_i;
        load(cl_i, gcl, {b, h, 0, i}, laneId);

        rt_fl<TB, TB> P, Ld, dM;
        zero(P);  mma_ABt(P, c_row, b_col, P);             // P = C_i·Bᵀ_j
        zero(Ld); add_row(Ld, Ld, cl_i); sub_col(Ld, Ld, cl_j); exp(Ld, Ld);
        zero(dM); mma_ABt(dM, dy_row, x_col, dM);          // dM = dY_i·Xᵀ_j
        rt_fl<TB, TB> M;
        mul(M, P, Ld);                                     // M = P⊙L
        if (i == j) { float zf = 0.0f; make_causal(M, M, laneId, zf); make_causal(dM, dM, laneId, zf); }
        rt_fl<TB, TB> dP;
        mul(dP, dM, Ld);                                   // dP = dM⊙L

        rt_bf<TB, TB> M_bf, dP_bf;
        copy(M_bf, M);   copy(dP_bf, dP);
        rt_bf<TB, TB, ducks::rt_layout::col> M_col, dP_col;
        swap_layout(M_col, M_bf, laneId);
        swap_layout(dP_col, dP_bf, laneId);
        mma_AtB(dx_reg, M_col, dy_row, dx_reg);            // dX_j += Mᵀ·dY_i  (j×D)
        mma_AtB(db_reg, dP_col, c_row, db_reg);            // dB_j += dPᵀ·C_i  (j×D)
    }
    store(gdB, db_reg, {b, h, j, 0}, laneId);
    store(gdX, dx_reg, {b, h, j, 0}, laneId);
}

#define instantiate_mamba2_bwd_i(D)                                            \
  template [[host_name("mamba2_bwd_i_" #D)]] [[kernel]] void mamba2_bwd_i<D>(   \
    device bf16* C [[buffer(0)]], device bf16* Bm [[buffer(1)]], device bf16* X [[buffer(2)]], \
    device bf16* dY [[buffer(3)]], device float* cl [[buffer(4)]], device bf16* dC [[buffer(5)]], \
    constant unsigned& N [[buffer(6)]], constant unsigned& H [[buffer(7)]],    \
    uint3 blockIdx [[threadgroup_position_in_grid]], uint laneId [[thread_index_in_simdgroup]]);
#define instantiate_mamba2_bwd_j(D)                                            \
  template [[host_name("mamba2_bwd_j_" #D)]] [[kernel]] void mamba2_bwd_j<D>(   \
    device bf16* C [[buffer(0)]], device bf16* Bm [[buffer(1)]], device bf16* X [[buffer(2)]], \
    device bf16* dY [[buffer(3)]], device float* cl [[buffer(4)]], device bf16* dB [[buffer(5)]], \
    device bf16* dX [[buffer(6)]], constant unsigned& N [[buffer(7)]], constant unsigned& H [[buffer(8)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], uint laneId [[thread_index_in_simdgroup]]);

instantiate_mamba2_bwd_i(64);  instantiate_mamba2_bwd_i(128);
instantiate_mamba2_bwd_j(64);  instantiate_mamba2_bwd_j(128);

}
