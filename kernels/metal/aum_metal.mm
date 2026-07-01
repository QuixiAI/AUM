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
static at::Tensor mamba2_mps(const at::Tensor& C_in, const at::Tensor& B_in,
                             const at::Tensor& X_in, const at::Tensor& cl_in) {
  TORCH_CHECK(C_in.device().is_mps(), "mamba2: tensors must be MPS");
  TORCH_CHECK(C_in.scalar_type() == at::kBFloat16, "mamba2: C,B,X must be bfloat16");
  TORCH_CHECK(cl_in.scalar_type() == at::kFloat, "mamba2: cumlog must be float32");
  auto C = C_in.contiguous(), B = B_in.contiguous(), X = X_in.contiguous(), cl = cl_in.contiguous();
  TORCH_CHECK(C.dim() == 4, "mamba2: C,B,X expect (B,H,N,D)");
  const int Bsz = C.size(0), H = C.size(1);
  const unsigned N = (unsigned)C.size(2);
  const int D = C.size(3);
  TORCH_CHECK(D == 64 || D == 128, "mamba2: D must be 64 or 128");
  TORCH_CHECK(N % 8 == 0, "mamba2: N must be a multiple of 8");
  auto out = at::empty_like(C);
  encode([&](Encoder& e) {
    e.pipeline("mamba2_" + std::to_string(D));
    e.in(C, 0); e.in(B, 1); e.in(X, 2); e.in(cl, 3); e.out(out, 4);
    e.bytes(N, 5); e.bytes((unsigned)H, 6);
    e.dispatch((int)N / 8, H, Bsz, 32, 1, 1);
  });
  return out;
}

// ---- mamba2 SSD backward -> (dC, dB, dX);  dcumlog is <dY,Y>-<dX,X> on the host ----
static std::tuple<at::Tensor, at::Tensor, at::Tensor> mamba2_bwd_mps(
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
  auto dC = at::empty_like(C), dB = at::empty_like(C), dX = at::empty_like(C);
  encode([&](Encoder& e) {                              // dC (fix query chunk i, loop j<=i)
    e.pipeline("mamba2_bwd_i_" + std::to_string(D));
    e.in(C, 0); e.in(B, 1); e.in(X, 2); e.in(dY, 3); e.in(cl, 4); e.out(dC, 5);
    e.bytes(N, 6); e.bytes((unsigned)H, 7);
    e.dispatch((int)N / 8, H, Bsz, 32, 1, 1);
  });
  encode([&](Encoder& e) {                              // dB, dX (fix key chunk j, loop i>=j)
    e.pipeline("mamba2_bwd_j_" + std::to_string(D));
    e.in(C, 0); e.in(B, 1); e.in(X, 2); e.in(dY, 3); e.in(cl, 4); e.out(dB, 5); e.out(dX, 6);
    e.bytes(N, 7); e.bytes((unsigned)H, 8);
    e.dispatch((int)N / 8, H, Bsz, 32, 1, 1);
  });
  return {dC, dB, dX};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_set_library", &aum_set_library, "set the metallib path");
  m.def("mamba2", &mamba2_mps, "AUM-Ø SSD forward (MPS)");
  m.def("mamba2_bwd", &mamba2_bwd_mps, "AUM-Ø SSD backward dC,dB,dX (MPS)");
}
