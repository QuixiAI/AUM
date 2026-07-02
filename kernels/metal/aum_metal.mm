// Self-contained PyTorch-MPS dispatch for the AUM-Ø Metal kernels (mamba2 SSD forward + backward).
//
// The compute lives in the .metal kernels in ./src (compiled to aum.metallib against the vendored
// MSL substrate in ./include). This file is the thin host glue that dispatches those kernels onto
// PyTorch's MPS stream. It has NO dependency on the ThunderMittens repo — the generic encoder
// adapter + tensor<->MTLBuffer plumbing (vendored from ThunderMittens' tk_torch pattern) and the
// per-kernel host ABI (buffer indices / grid) are both inlined below.

#include <torch/extension.h>
#include <torch/mps.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string>
#include <tuple>
#include <unordered_map>

// The MTLBuffer backing an MPS tensor's storage (documented PyTorch pattern).
static inline id<MTLBuffer> mtl_buffer(const at::Tensor& t) {
  return __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
}
static inline NSUInteger byte_offset(const at::Tensor& t) {
  return static_cast<NSUInteger>(t.storage_offset()) * t.element_size();
}

// ---- lazily-loaded metallib + pipeline-state cache (keyed by function name) ----
static std::string g_metallib_path;
static id<MTLLibrary> g_library = nil;
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_pipelines;

static void aum_set_library(const std::string& path) {
  g_metallib_path = path;
  g_library = nil;
  g_pipelines.clear();
}

static id<MTLComputePipelineState> aum_pipeline(id<MTLDevice> device, NSString* name) {
  std::string key = name.UTF8String;
  auto it = g_pipelines.find(key);
  if (it != g_pipelines.end()) return it->second;
  NSError* err = nil;
  if (g_library == nil) {
    TORCH_CHECK(!g_metallib_path.empty(), "aum_metal: metallib path not set; call _set_library() first");
    NSString* p = [NSString stringWithUTF8String:g_metallib_path.c_str()];
    g_library = [device newLibraryWithURL:[NSURL fileURLWithPath:p] error:&err];
    TORCH_CHECK(g_library != nil, "aum_metal: failed to load metallib at ", g_metallib_path);
  }
  id<MTLFunction> fn = [g_library newFunctionWithName:name];
  TORCH_CHECK(fn != nil, "aum_metal: kernel function not found: ", name.UTF8String);
  id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
  TORCH_CHECK(pso != nil, "aum_metal: failed to create pipeline for ", name.UTF8String);
  g_pipelines[key] = pso;
  return pso;
}

// ---- encoder adapter ----
struct Encoder {
  id<MTLComputeCommandEncoder> enc;
  id<MTLDevice> device;
  void pipeline(const std::string& name) {
    [enc setComputePipelineState:aum_pipeline(device, [NSString stringWithUTF8String:name.c_str()])];
  }
  void in(const at::Tensor& t, int i) { [enc setBuffer:mtl_buffer(t) offset:byte_offset(t) atIndex:i]; }
  void out(const at::Tensor& t, int i) { [enc setBuffer:mtl_buffer(t) offset:byte_offset(t) atIndex:i]; }
  template <class T> void bytes(const T& v, int i) { [enc setBytes:&v length:sizeof(T) atIndex:i]; }
  void dispatch(int gx, int gy, int gz, int tx, int ty, int tz) {
    [enc dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
  }
};

// Run fn(encoder) on torch's MPS stream (committed at the next stream sync).
template <class F>
static void encode(F fn) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
    dispatch_queue_t q = torch::mps::get_dispatch_queue();
    id<MTLDevice> dev = cb.device;
    dispatch_sync(q, ^{
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      Encoder e{enc, dev};
      fn(e);
      [enc endEncoding];
    });
  }
}

// ---- mamba2 SSD forward:  Y = ((C@Bᵀ) ⊙ exp(cl_i−cl_j) ⊙ causal) @ X ----
// Two routes over one math: the quadratic materialized kernel, and the chunked linear-time
// 3-kernel pipeline (ssd_chunk_kv -> scan -> out, O(N·(L+D)·D)) whose DxD chunk state is
// quadrant-tiled into 64x64 register blocks (kv grid gains a QB*QB axis, QB = D/64; the scanned
// state is stored bf16 — the out mma consumes bf16 anyway, so results are identical and the
// dominant state-read traffic halves). Auto-route thresholds are MEASURED (M-series):
// chunked wins from N>=2048 at D=64 (2-7x) and N>=8192 at D=128 (2.1x; parity at 4096 — the
// remaining gap is per-query-tile state reloads, the cooperative-sharing follow-up).
static void mamba2_check(const at::Tensor& C, const at::Tensor& cl) {
  TORCH_CHECK(C.device().is_mps(), "mamba2: tensors must be MPS");
  TORCH_CHECK(C.scalar_type() == at::kBFloat16, "mamba2: C,B,X must be bfloat16");
  TORCH_CHECK(cl.scalar_type() == at::kFloat, "mamba2: cumlog must be float32");
  TORCH_CHECK(C.dim() == 4, "mamba2: C,B,X expect (B,H,N,D)");
  const int D = C.size(3);
  TORCH_CHECK(D == 64 || D == 128, "mamba2: D must be 64 or 128");
  TORCH_CHECK(C.size(2) % 8 == 0, "mamba2: N must be a multiple of 8");
}

static at::Tensor mamba2_quadratic(const at::Tensor& C, const at::Tensor& B,
                                   const at::Tensor& X, const at::Tensor& cl) {
  const int Bsz = C.size(0), H = C.size(1), D = C.size(3);
  const unsigned N = (unsigned)C.size(2);
  auto out = at::empty_like(C);
  encode([&](Encoder& e) {
    e.pipeline("mamba2_" + std::to_string(D));
    e.in(C, 0); e.in(B, 1); e.in(X, 2); e.in(cl, 3); e.out(out, 4);
    e.bytes(N, 5); e.bytes((unsigned)H, 6);
    e.dispatch((int)N / 8, H, Bsz, 32, 1, 1);
  });
  return out;
}

static constexpr unsigned kSsdChunkL = 64;         // must match SSD_CHUNK_L in the metal

static at::Tensor mamba2_chunked(const at::Tensor& C, const at::Tensor& B,
                                 const at::Tensor& X, const at::Tensor& cl) {
  const int Bsz = C.size(0), H = C.size(1), D = C.size(3);
  const unsigned N = (unsigned)C.size(2);
  TORCH_CHECK(N % kSsdChunkL == 0 && N >= 2 * kSsdChunkL,
              "mamba2_chunked: N must be a multiple of 64 and >= 128");
  const int Cn = (int)(N / kSsdChunkL);
  const int QB = D / 64;                           // 64x64 state quadrants per side
  auto out = at::empty_like(C);
  auto s_raw = at::empty({Bsz, H, Cn, D, D}, C.options().dtype(at::kFloat));
  auto s_ex = at::empty({Bsz, H, Cn, D, D}, C.options());   // bf16 (see routing note above)
  encode([&](Encoder& e) {
    e.pipeline("ssd_chunk_kv_" + std::to_string(D));
    e.in(B, 0); e.in(X, 1); e.in(cl, 2); e.out(s_raw, 3);
    e.bytes(N, 4); e.bytes((unsigned)H, 5);
    e.dispatch(Cn * QB * QB, H, Bsz, 32, 1, 1);
  });
  encode([&](Encoder& e) {
    e.pipeline("ssd_chunk_scan_" + std::to_string(D));
    e.in(s_raw, 0); e.in(cl, 1); e.out(s_ex, 2);
    e.bytes((unsigned)Cn, 3); e.bytes(N, 4);
    e.dispatch(Bsz * H, 1, 1, 256, 1, 1);
  });
  encode([&](Encoder& e) {
    e.pipeline("ssd_chunk_out_" + std::to_string(D));
    e.in(C, 0); e.in(B, 1); e.in(X, 2); e.in(cl, 3); e.in(s_ex, 4); e.out(out, 5);
    e.bytes(N, 6); e.bytes((unsigned)H, 7);
    e.dispatch(Cn, H, Bsz, 256, 1, 1);           // cooperative: one threadgroup per chunk
  });
  return out;
}

static at::Tensor mamba2_mps(const at::Tensor& C_in, const at::Tensor& B_in,
                             const at::Tensor& X_in, const at::Tensor& cl_in) {
  auto C = C_in.contiguous(), B = B_in.contiguous(), X = X_in.contiguous(), cl = cl_in.contiguous();
  mamba2_check(C, cl);
  const unsigned N = (unsigned)C.size(2);
  const int D = C.size(3);
  const unsigned chunk_min = (D == 64) ? 2048 : 4096;      // measured crossovers (coop)
  if (N % kSsdChunkL != 0 || N < chunk_min)
    return mamba2_quadratic(C, B, X, cl);
  return mamba2_chunked(C, B, X, cl);
}

static at::Tensor mamba2_chunked_mps(const at::Tensor& C_in, const at::Tensor& B_in,
                                     const at::Tensor& X_in, const at::Tensor& cl_in) {
  auto C = C_in.contiguous(), B = B_in.contiguous(), X = X_in.contiguous(), cl = cl_in.contiguous();
  mamba2_check(C, cl);
  return mamba2_chunked(C, B, X, cl);                      // forced route (tests / benchmarks)
}

// ---- mamba2 SSD backward -> (dC, dB, dX, dcumlog), two routes like the forward ----
// Quadratic: mamba2_bwd_row/col with IN-KERNEL fp32 dcumlog = rowsum(M) - colsum(M), M = dSt∘S.
// Chunked (linear-time): recompute the forward chunk states, build the gradient states G_c,
// reverse-scan dKV, then chunk-bounded row/col kernels; dcumlog via the exact identity
// dcl = <dY,Y> - <dX,X> with a linear-time Y recompute.
static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mamba2_bwd_quadratic(
    const at::Tensor& C, const at::Tensor& B, const at::Tensor& X,
    const at::Tensor& cl, const at::Tensor& dY) {
  const int Bsz = C.size(0), H = C.size(1), D = C.size(3);
  const unsigned N = (unsigned)C.size(2);
  auto f32 = C.options().dtype(at::kFloat);
  auto dC = at::empty_like(C), dB = at::empty_like(C), dX = at::empty_like(C);
  auto r = at::empty({Bsz, H, (long)N}, f32);
  auto cc = at::empty({Bsz, H, (long)N}, f32);
  encode([&](Encoder& e) {                              // dC + rowsum(M) (fix row tile, loop j<=i)
    e.pipeline("mamba2_bwd_row_" + std::to_string(D));
    e.in(C, 0); e.in(B, 1); e.in(X, 2); e.in(cl, 3); e.in(dY, 4); e.out(dC, 5); e.out(r, 6);
    e.bytes(N, 7); e.bytes((unsigned)H, 8);
    e.dispatch((int)N / 8, H, Bsz, 32, 1, 1);
  });
  encode([&](Encoder& e) {                              // dB, dX + colsum(M) (fix col tile, loop i>=j)
    e.pipeline("mamba2_bwd_col_" + std::to_string(D));
    e.in(C, 0); e.in(B, 1); e.in(X, 2); e.in(cl, 3); e.in(dY, 4);
    e.out(dB, 5); e.out(dX, 6); e.out(cc, 7);
    e.bytes(N, 8); e.bytes((unsigned)H, 9);
    e.dispatch((int)N / 8, H, Bsz, 32, 1, 1);
  });
  return {dC, dB, dX, r - cc};
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mamba2_bwd_chunked(
    const at::Tensor& C, const at::Tensor& B, const at::Tensor& X,
    const at::Tensor& cl, const at::Tensor& dY) {
  const int Bsz = C.size(0), H = C.size(1), D = C.size(3);
  const unsigned N = (unsigned)C.size(2);
  TORCH_CHECK(N % kSsdChunkL == 0 && N >= 2 * kSsdChunkL,
              "mamba2_bwd_chunked: N must be a multiple of 64 and >= 128");
  const int Cn = (int)(N / kSsdChunkL);
  const int QB = D / 64;
  auto f32 = C.options().dtype(at::kFloat);
  auto dC = at::empty_like(C), dB = at::empty_like(C), dX = at::empty_like(C);
  auto s_raw = at::empty({Bsz, H, Cn, D, D}, f32);
  auto s_ex = at::empty({Bsz, H, Cn, D, D}, C.options());          // bf16
  auto dkv = at::empty({Bsz, H, Cn, D, D}, C.options());           // bf16
  auto r = at::empty({Bsz, H, (long)N}, f32), ri = at::empty({Bsz, H, (long)N}, f32);
  auto cc = at::empty({Bsz, H, (long)N}, f32), ci = at::empty({Bsz, H, (long)N}, f32);
  encode([&](Encoder& e) {                              // forward chunk states (recompute)
    e.pipeline("ssd_chunk_kv_" + std::to_string(D));
    e.in(B, 0); e.in(X, 1); e.in(cl, 2); e.out(s_raw, 3);
    e.bytes(N, 4); e.bytes((unsigned)H, 5);
    e.dispatch(Cn * QB * QB, H, Bsz, 32, 1, 1);
  });
  encode([&](Encoder& e) {
    e.pipeline("ssd_chunk_scan_" + std::to_string(D));
    e.in(s_raw, 0); e.in(cl, 1); e.out(s_ex, 2);
    e.bytes((unsigned)Cn, 3); e.bytes(N, 4);
    e.dispatch(Bsz * H, 1, 1, 256, 1, 1);
  });
  encode([&](Encoder& e) {                              // G_c = dSex_c (reuses s_raw: dead after scan)
    e.pipeline("ssd_chunk_gstate_" + std::to_string(D));
    e.in(C, 0); e.in(dY, 1); e.in(cl, 2); e.out(s_raw, 3);
    e.bytes(N, 4); e.bytes((unsigned)H, 5);
    e.dispatch(Cn * QB * QB, H, Bsz, 32, 1, 1);
  });
  encode([&](Encoder& e) {                              // dKV: reverse decayed suffix
    e.pipeline("ssd_chunk_rscan_" + std::to_string(D));
    e.in(s_raw, 0); e.in(cl, 1); e.out(dkv, 2);
    e.bytes((unsigned)Cn, 3); e.bytes(N, 4);
    e.dispatch(Bsz * H, 1, 1, 256, 1, 1);
  });
  encode([&](Encoder& e) {                              // dC + intra/inter rowsum(M)
    e.pipeline("ssd_chunk_bwd_row_" + std::to_string(D));
    e.in(C, 0); e.in(B, 1); e.in(X, 2); e.in(cl, 3); e.in(dY, 4); e.in(s_ex, 5); e.out(dC, 6);
    e.out(r, 7); e.out(ri, 8);
    e.bytes(N, 9); e.bytes((unsigned)H, 10);
    e.dispatch(Cn, H, Bsz, 256, 1, 1);           // cooperative
  });
  encode([&](Encoder& e) {                              // dB, dX + intra/inter colsum(M)
    e.pipeline("ssd_chunk_bwd_col_" + std::to_string(D));
    e.in(C, 0); e.in(B, 1); e.in(X, 2); e.in(cl, 3); e.in(dY, 4); e.in(dkv, 5);
    e.out(dB, 6); e.out(dX, 7); e.out(cc, 8); e.out(ci, 9);
    e.bytes(N, 10); e.bytes((unsigned)H, 11);
    e.dispatch(Cn, H, Bsz, 256, 1, 1);           // cooperative
  });
  // In-kernel split dcl (fp32, no Y recompute): rowsum(M) = r_intra + <dC_inter, C_i>,
  // colsum(M) = cc_intra + <dX_inter, X_j>.
  return {dC, dB, dX, (r + ri) - (cc + ci)};
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mamba2_bwd_mps(
    const at::Tensor& C_in, const at::Tensor& B_in, const at::Tensor& X_in,
    const at::Tensor& cl_in, const at::Tensor& dY_in) {
  TORCH_CHECK(dY_in.scalar_type() == at::kBFloat16, "mamba2_bwd: dY must be bfloat16");
  auto C = C_in.contiguous(), B = B_in.contiguous(), X = X_in.contiguous(),
       cl = cl_in.contiguous(), dY = dY_in.contiguous();
  mamba2_check(C, cl);
  const unsigned N = (unsigned)C.size(2);
  const int D = C.size(3);
  const unsigned chunk_min = (D == 64) ? 2048 : 4096;      // measured crossovers (coop)
  if (N % kSsdChunkL != 0 || N < chunk_min)
    return mamba2_bwd_quadratic(C, B, X, cl, dY);
  return mamba2_bwd_chunked(C, B, X, cl, dY);
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mamba2_bwd_chunked_mps(
    const at::Tensor& C_in, const at::Tensor& B_in, const at::Tensor& X_in,
    const at::Tensor& cl_in, const at::Tensor& dY_in) {
  auto C = C_in.contiguous(), B = B_in.contiguous(), X = X_in.contiguous(),
       cl = cl_in.contiguous(), dY = dY_in.contiguous();
  mamba2_check(C, cl);
  return mamba2_bwd_chunked(C, B, X, cl, dY);              // forced route (tests / benchmarks)
}

// ---- aum_decode: single-token U-phase step  S <- a*S + x⊗k_rot ; out = S·q_rot ----
// S (B,H,D,D) is updated in place and returned; out (B,H,D) is fresh. All fp32.
static std::tuple<at::Tensor, at::Tensor> aum_decode_mps(
    const at::Tensor& S_in, const at::Tensor& a_in, const at::Tensor& x_in,
    const at::Tensor& k_in, const at::Tensor& q_in) {
  TORCH_CHECK(S_in.device().is_mps(), "aum_decode: tensors must be MPS");
  TORCH_CHECK(S_in.scalar_type() == at::kFloat && a_in.scalar_type() == at::kFloat &&
              x_in.scalar_type() == at::kFloat && k_in.scalar_type() == at::kFloat &&
              q_in.scalar_type() == at::kFloat, "aum_decode: all inputs must be float32");
  auto S = S_in.contiguous(), a = a_in.contiguous(), x = x_in.contiguous(),
       k = k_in.contiguous(), q = q_in.contiguous();
  TORCH_CHECK(S.dim() == 4, "aum_decode: S expects (B,H,D,D)");
  const int Bsz = S.size(0), H = S.size(1), D = S.size(2);
  TORCH_CHECK(S.size(3) == D, "aum_decode: S must be square (Dv==Dqk)");
  TORCH_CHECK(D == 64 || D == 128, "aum_decode: D must be 64 or 128");
  auto out = at::empty({Bsz, H, D}, S.options());
  encode([&](Encoder& e) {
    e.pipeline("aum_decode_" + std::to_string(D));
    e.in(S, 0); e.in(a, 1); e.in(x, 2); e.in(k, 3); e.in(q, 4); e.out(out, 5);
    e.bytes((unsigned)H, 6);
    e.dispatch(1, H, Bsz, D, 1, 1);            // one threadgroup per (b,h); D threads (one per row)
  });
  return {out, S};
}

// ---- AUM fused U-phase operand pipeline (§4) ----
// aum_operands: one pass builds the SSD operands C = R(phi)q, B = R(phi)k_hat, X = rho*tau*v_hat
// from the model's (B, N, H, Dh) layout into the kernel's (B, H, N, D) layout (transpose free).
static std::tuple<at::Tensor, at::Tensor, at::Tensor> aum_operands_mps(
    const at::Tensor& q_in, const at::Tensor& k_in, const at::Tensor& v_in,
    const at::Tensor& phi_in, const at::Tensor& rtw_in, const at::Tensor& freqs_in, double eps) {
  TORCH_CHECK(q_in.device().is_mps() && q_in.scalar_type() == at::kBFloat16,
              "aum_operands: q,k,v must be bf16 MPS");
  TORCH_CHECK(phi_in.scalar_type() == at::kFloat && rtw_in.scalar_type() == at::kFloat &&
              freqs_in.scalar_type() == at::kFloat, "aum_operands: phi/rtw/freqs must be fp32");
  auto q = q_in.contiguous(), k = k_in.contiguous(), v = v_in.contiguous();
  auto phi = phi_in.contiguous(), rtw = rtw_in.contiguous(), freqs = freqs_in.contiguous();
  TORCH_CHECK(q.dim() == 4, "aum_operands: q,k,v expect (B, N, H, Dh)");
  const int Bsz = q.size(0), H = q.size(2), D = q.size(3);
  const unsigned N = (unsigned)q.size(1);
  TORCH_CHECK(D % 64 == 0, "aum_operands: Dh must be a multiple of 64");
  auto C = at::empty({Bsz, H, (long)N, D}, q.options());
  auto B = at::empty_like(C), X = at::empty_like(C);
  const float epsf = (float)eps;
  encode([&](Encoder& e) {
    e.pipeline("aum_operands");
    e.in(q, 0); e.in(k, 1); e.in(v, 2); e.in(phi, 3); e.in(rtw, 4); e.in(freqs, 5);
    e.out(C, 6); e.out(B, 7); e.out(X, 8);
    e.bytes(N, 9); e.bytes((unsigned)H, 10); e.bytes((unsigned)D, 11); e.bytes(epsf, 12);
    e.dispatch((int)N, H, Bsz, 32, 1, 1);          // one simdgroup per (b, l, h) row
  });
  return {C, B, X};
}

// aum_epilogue: out = silu(z) * RMSNorm(Y + Dskip*v) * w_norm, writing the model layout back.
static at::Tensor aum_epilogue_mps(const at::Tensor& Y_in, const at::Tensor& v_in,
                                   const at::Tensor& z_in, const at::Tensor& dsk_in,
                                   const at::Tensor& wn_in, double eps) {
  TORCH_CHECK(Y_in.device().is_mps() && Y_in.scalar_type() == at::kBFloat16,
              "aum_epilogue: Y,v,z must be bf16 MPS");
  TORCH_CHECK(dsk_in.scalar_type() == at::kFloat && wn_in.scalar_type() == at::kFloat,
              "aum_epilogue: Dskip / norm weight must be fp32");
  auto Y = Y_in.contiguous(), v = v_in.contiguous(), z = z_in.contiguous();
  auto dsk = dsk_in.contiguous(), wn = wn_in.contiguous();
  TORCH_CHECK(v.dim() == 4, "aum_epilogue: v,z expect (B, N, H, Dh)");
  const int Bsz = v.size(0), H = v.size(2), D = v.size(3);
  const unsigned N = (unsigned)v.size(1);
  auto out = at::empty_like(v);
  const float epsf = (float)eps;
  encode([&](Encoder& e) {
    e.pipeline("aum_epilogue");
    e.in(Y, 0); e.in(v, 1); e.in(z, 2); e.in(dsk, 3); e.in(wn, 4); e.out(out, 5);
    e.bytes(N, 6); e.bytes((unsigned)H, 7); e.bytes((unsigned)D, 8); e.bytes(epsf, 9);
    e.dispatch((int)N, H, Bsz, 32, 1, 1);
  });
  return out;
}

// ---- fused cross-entropy over the vocab axis (vendored from ThunderMittens) ----
// One simdgroup per row striding V; never materializes probabilities. fwd -> (loss, lse);
// bwd recomputes p = exp(x - lse) and scales by per-row grad_out. The `mw` variants use 4
// simdgroups per row (for small T / large V).
static std::string ce_type(const at::Tensor& t) {
  if (t.scalar_type() == at::kFloat) return "float32";
  if (t.scalar_type() == at::kHalf) return "float16";
  TORCH_CHECK(t.scalar_type() == at::kBFloat16, "cross_entropy: logits must be f32/f16/bf16");
  return "bfloat16";
}

static std::tuple<at::Tensor, at::Tensor> cross_entropy_fwd_mps(
    const at::Tensor& logits_in, const at::Tensor& targets_in, int64_t ignore_index,
    double label_smoothing, double z_loss, double softcap) {
  TORCH_CHECK(logits_in.device().is_mps() && logits_in.dim() == 2, "cross_entropy: logits (T,V) MPS");
  TORCH_CHECK(targets_in.scalar_type() == at::kInt || targets_in.scalar_type() == at::kLong,
              "cross_entropy: int targets");
  auto logits = logits_in.contiguous();
  auto targets = targets_in.to(at::kInt).contiguous();
  const int T = logits.size(0), V = logits.size(1);
  auto f32 = logits.options().dtype(at::kFloat);
  auto loss = at::empty({T}, f32), lse = at::empty({T}, f32);
  const bool mw = T < 1024;                       // small-T route: 4 simdgroups per row
  const float ls = (float)label_smoothing, zl = (float)z_loss, sc = (float)softcap;
  encode([&](Encoder& e) {
    e.pipeline((mw ? "cross_entropy_fwd_mw_" : "cross_entropy_fwd_") + ce_type(logits));
    e.in(logits, 0); e.in(targets, 1); e.out(loss, 2); e.out(lse, 3);
    e.bytes(V, 4); e.bytes((int)ignore_index, 5); e.bytes(ls, 6); e.bytes(zl, 7); e.bytes(sc, 8);
    e.dispatch(T, 1, 1, mw ? 128 : 32, 1, 1);
  });
  return {loss, lse};
}

// ---- aum_silence: the fused sequential global block (v6 §5-§9; roadmap step 4) ----
// Marches the whole silence recurrence over L tokens in ONE kernel launch: one threadgroup per
// batch row, 256 threads cooperating per token. Geometry hardcoded in the metal (D=512, DS=128,
// DM=32, H=8, DH=64, J=2). Stream/weight/save pack layouts are defined by
// aum_ssm/ops/metal/silence_flat.py + aum_ssm/ops/metal/silence_metal.py.
static constexpr int kSilSW = 2720;                 // stream-pack width  (must match SW)
static constexpr int kSilSV = 3151;                 // save-pack width    (must match SV)

static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> aum_silence_fwd_mps(
    const at::Tensor& streams_in, const at::Tensor& alpha_in, const at::Tensor& xw_in,
    const at::Tensor& krot_in, const at::Tensor& haltu_in, const at::Tensor& wpack_in,
    double kappa, int64_t forced) {
  auto streams = streams_in.contiguous(), alpha = alpha_in.contiguous();
  auto xw = xw_in.contiguous(), krot = krot_in.contiguous();
  auto haltu = haltu_in.contiguous(), wpack = wpack_in.contiguous();
  TORCH_CHECK(streams.device().is_mps() && streams.scalar_type() == at::kFloat,
              "aum_silence_fwd: fp32 MPS tensors required");
  const int Bsz = streams.size(0);
  const unsigned L = (unsigned)streams.size(1);
  TORCH_CHECK(streams.size(2) == kSilSW, "aum_silence_fwd: stream pack width mismatch");
  auto f32 = streams.options();
  auto S = at::zeros({Bsz, 8, 64, 64}, f32);
  const long nseg = (L + 63) / 64;
  auto S_ckpt = at::empty({Bsz, nseg, 8, 64, 64}, f32);
  auto save = at::empty({Bsz, (long)L, kSilSV}, f32);
  auto jstar = at::empty({Bsz, (long)L}, streams.options().dtype(at::kInt));
  const float kap = (float)kappa;
  const int frc = (int)forced;
  encode([&](Encoder& e) {
    e.pipeline("aum_silence_fwd");
    e.in(streams, 0); e.in(alpha, 1); e.in(xw, 2); e.in(krot, 3); e.in(haltu, 4);
    e.in(wpack, 5); e.out(S, 6); e.out(save, 7); e.out(jstar, 8); e.out(S_ckpt, 9);
    e.bytes(L, 10); e.bytes(kap, 11); e.bytes(frc, 12);
    e.dispatch(1, 1, Bsz, 256, 1, 1);                // one threadgroup per batch row
  });
  return {save, jstar, S, S_ckpt};
}

static at::Tensor cross_entropy_bwd_mps(
    const at::Tensor& logits_in, const at::Tensor& targets_in, const at::Tensor& lse_in,
    const at::Tensor& grad_out_in, int64_t ignore_index, double label_smoothing,
    double z_loss, double softcap) {
  auto logits = logits_in.contiguous();
  auto targets = targets_in.to(at::kInt).contiguous();
  auto lse = lse_in.contiguous(), go = grad_out_in.to(at::kFloat).contiguous();
  const int T = logits.size(0), V = logits.size(1);
  auto grad = at::empty_like(logits);
  const bool mw = T < 1024;
  const float ls = (float)label_smoothing, zl = (float)z_loss, sc = (float)softcap;
  encode([&](Encoder& e) {
    e.pipeline((mw ? "cross_entropy_bwd_mw_" : "cross_entropy_bwd_") + ce_type(logits));
    e.in(logits, 0); e.in(targets, 1); e.in(lse, 2); e.in(go, 3); e.out(grad, 4);
    e.bytes(V, 5); e.bytes((int)ignore_index, 6); e.bytes(ls, 7); e.bytes(zl, 8); e.bytes(sc, 9);
    e.dispatch(T, 1, 1, mw ? 128 : 32, 1, 1);
  });
  return grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_set_library", &aum_set_library, "set the metallib path");
  m.def("mamba2", &mamba2_mps, "AUM-Ø SSD forward (MPS, auto-routed)");
  m.def("mamba2_chunked", &mamba2_chunked_mps, "AUM-Ø SSD forward, forced chunked route");
  m.def("mamba2_bwd", &mamba2_bwd_mps, "AUM-Ø SSD backward (MPS, auto-routed)");
  m.def("mamba2_bwd_chunked", &mamba2_bwd_chunked_mps, "AUM-Ø SSD backward, forced chunked route");
  m.def("aum_decode", &aum_decode_mps, "AUM-Ø single-token U-phase decode step (MPS)");
  m.def("aum_silence_fwd", &aum_silence_fwd_mps,
        "AUM-Ø fused sequential global block, forward (MPS)");
  m.def("aum_operands", &aum_operands_mps, "AUM-Ø fused U-phase operand builder (MPS)");
  m.def("aum_epilogue", &aum_epilogue_mps, "AUM-Ø fused U-phase epilogue (MPS)");
  m.def("cross_entropy_fwd", &cross_entropy_fwd_mps, "fused CE forward -> (loss, lse) (MPS)");
  m.def("cross_entropy_bwd", &cross_entropy_bwd_mps, "fused CE backward -> grad_logits (MPS)");
}
