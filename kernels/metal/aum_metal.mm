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
    e.dispatch((int)N / 8, H, Bsz, 32, 1, 1);
  });
  return out;
}

static at::Tensor mamba2_mps(const at::Tensor& C_in, const at::Tensor& B_in,
                             const at::Tensor& X_in, const at::Tensor& cl_in) {
  auto C = C_in.contiguous(), B = B_in.contiguous(), X = X_in.contiguous(), cl = cl_in.contiguous();
  mamba2_check(C, cl);
  const unsigned N = (unsigned)C.size(2);
  const int D = C.size(3);
  const unsigned chunk_min = (D == 64) ? 2048 : 8192;      // measured crossovers
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

// ---- mamba2 SSD backward -> (dC, dB, dX, dcumlog) ----
// dcumlog = rowsum(M) - colsum(M), M = dSt ∘ S, accumulated IN-KERNEL in fp32 (mamba2_bwd_row
// emits r, mamba2_bwd_col emits cc) — no host identity, and the forward output Y is not needed.
static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mamba2_bwd_mps(
    const at::Tensor& C_in, const at::Tensor& B_in, const at::Tensor& X_in,
    const at::Tensor& cl_in, const at::Tensor& dY_in) {
  TORCH_CHECK(C_in.device().is_mps(), "mamba2_bwd: tensors must be MPS");
  TORCH_CHECK(C_in.scalar_type() == at::kBFloat16 && dY_in.scalar_type() == at::kBFloat16,
              "mamba2_bwd: C,B,X,dY must be bfloat16");
  TORCH_CHECK(cl_in.scalar_type() == at::kFloat, "mamba2_bwd: cumlog must be float32");
  auto C = C_in.contiguous(), B = B_in.contiguous(), X = X_in.contiguous(),
       cl = cl_in.contiguous(), dY = dY_in.contiguous();
  const int Bsz = C.size(0), H = C.size(1);
  const unsigned N = (unsigned)C.size(2);
  const int D = C.size(3);
  TORCH_CHECK(D == 64 || D == 128, "mamba2_bwd: D must be 64 or 128");
  TORCH_CHECK(N % 8 == 0, "mamba2_bwd: N must be a multiple of 8");
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_set_library", &aum_set_library, "set the metallib path");
  m.def("mamba2", &mamba2_mps, "AUM-Ø SSD forward (MPS, auto-routed)");
  m.def("mamba2_chunked", &mamba2_chunked_mps, "AUM-Ø SSD forward, forced chunked route");
  m.def("mamba2_bwd", &mamba2_bwd_mps, "AUM-Ø SSD backward dC,dB,dX (MPS)");
  m.def("aum_decode", &aum_decode_mps, "AUM-Ø single-token U-phase decode step (MPS)");
}
