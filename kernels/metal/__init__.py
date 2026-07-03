"""Self-contained PyTorch-MPS build for the AUM-Ø Metal kernels — no ThunderMittens repo needed.

Builds independently from source in THIS tree:
  - MSL substrate (tile primitives) is vendored in ./include,
  - the AUM kernels are in ./src (mamba2.metal SSD forward, mamba2_bwd.metal SSD backward),
  - ./aum_metal.mm is the torch-MPS dispatch (generic encoder + our two kernels' host ABI).

On import this compiles ./src/*.metal into aum.metallib with `xcrun metal` and JIT-builds the
ObjC++ dispatch via torch.utils.cpp_extension.load. Requirements: PyTorch (MPS) + Xcode's Metal
toolchain (`xcrun metal`). See NOTICE for attribution of the vendored substrate.
"""

import os
import subprocess

import torch  # noqa: F401  (extension links against torch)
from torch.utils.cpp_extension import load

if not torch.backends.mps.is_available():
    # ImportError (not FileNotFoundError from xcrun) so pytest.importorskip and the model's
    # fused-route eligibility try/except treat non-Mac nodes as "backend unavailable".
    raise ImportError("kernels.metal requires PyTorch MPS (Apple GPU); "
                      "on CUDA use kernels.triton")

_HERE = os.path.dirname(os.path.abspath(__file__))
_INCLUDE = os.path.join(_HERE, "include")
_SRC = os.path.join(_HERE, "src")
_METALLIB = os.path.join(_HERE, "aum.metallib")
_METAL_SOURCES = [os.path.join(_SRC, "mamba2.metal"), os.path.join(_SRC, "aum_decode.metal"),
                  os.path.join(_SRC, "aum_unfold.metal"), os.path.join(_SRC, "cross_entropy.metal"),
                  os.path.join(_SRC, "aum_silence.metal")]


def build_metallib(force: bool = False) -> str:
    """Compile ./src/*.metal into aum.metallib via `xcrun metal` (rebuild only if stale)."""
    if not force and os.path.exists(_METALLIB):
        if os.path.getmtime(_METALLIB) >= max(os.path.getmtime(s) for s in _METAL_SOURCES):
            return _METALLIB
    subprocess.run(["xcrun", "metal", "-std=metal3.1", "-O2", "-I", _INCLUDE,
                    *_METAL_SOURCES, "-o", _METALLIB], check=True)
    return _METALLIB


build_metallib()
_ext = load(
    name="aum_metal_ext",
    sources=[os.path.join(_HERE, "aum_metal.mm")],
    extra_cflags=["-std=c++17"],
    extra_ldflags=["-framework", "Metal", "-framework", "Foundation", "-framework", "QuartzCore"],
    verbose=False,
)
_ext._set_library(_METALLIB)


def mamba2(C, B, X, cumlog):
    """SSD forward ((C@Bᵀ)⊙exp(cl_i−cl_j)⊙causal)@X. C,B,X bf16 (B,H,N,D); cumlog fp32 (B,H,N).
    MPS; D in {64,128}, N%8. Auto-routed between the quadratic kernel and the chunked LINEAR-TIME
    pipeline (64x64 quadrant-tiled state, both head dims) at the MEASURED crossovers:
    N>=2048 for D=64, N>=8192 for D=128 (N%64==0 required for chunked)."""
    return _ext.mamba2(C, B, X, cumlog)


def mamba2_chunked(C, B, X, cumlog):
    """The chunked linear-time route, forced (testing/benchmarks). Requires N%64==0, N>=128."""
    return _ext.mamba2_chunked(C, B, X, cumlog)


def mamba2_bwd(C, B, X, cumlog, dY):
    """SSD backward -> (dC, dB, dX, dcumlog). C,B,X,dY bf16 (B,H,N,D); cumlog fp32 (B,H,N). MPS;
    D in {64,128}. Auto-routed like the forward: the chunked LINEAR-TIME backward (gradient
    states + reverse decayed scan + chunk-bounded tiles; dcumlog via the exact <dY,Y>-<dX,X>
    identity over a linear-time Y recompute) above the measured crossovers, the quadratic
    row/col kernels (in-kernel fp32 dcumlog) otherwise."""
    return _ext.mamba2_bwd(C, B, X, cumlog, dY)


def mamba2_bwd_chunked(C, B, X, cumlog, dY):
    """The chunked linear-time backward, forced (testing/benchmarks). N%64==0, N>=128."""
    return _ext.mamba2_bwd_chunked(C, B, X, cumlog, dY)


def aum_operands(q, k, v, phi, rho_tau, freqs, eps=1e-6):
    """Fused U-phase operand builder (§4): C = R(phi)q, B = R(phi)(k/||k||), X = rho*tau*(v/||v||),
    reading (B,N,H,Dh) bf16 and writing the kernel layout (B,H,N,D). phi/rho_tau (B,N,H) fp32;
    freqs (Dh/2) fp32. One pass — replaces the host rotation/norm/scale/rearrange chain."""
    return _ext.aum_operands(q, k, v, phi, rho_tau, freqs, float(eps))


def aum_epilogue(Y, v, z, d_skip, norm_weight, eps=1e-5):
    """Fused U-phase epilogue: silu(z) * RMSNorm(Y + d_skip*v) * norm_weight. Y (B,H,N,D) bf16;
    v,z (B,N,H,Dh) bf16; d_skip flat (H*Dh) fp32; norm_weight (Dh) fp32. Writes (B,N,H,Dh)."""
    return _ext.aum_epilogue(Y, v, z, d_skip, norm_weight, float(eps))


def cross_entropy_fwd(logits, targets, ignore_index=-100, label_smoothing=0.0, z_loss=0.0,
                      softcap=0.0):
    """Fused CE forward over the vocab axis -> (loss (T,), lse (T,)) fp32; logits (T,V) f32/f16/
    bf16 MPS. Never materializes probabilities (vendored from ThunderMittens)."""
    return _ext.cross_entropy_fwd(logits, targets, int(ignore_index), float(label_smoothing),
                                  float(z_loss), float(softcap))


def cross_entropy_bwd(logits, targets, lse, grad_out, ignore_index=-100, label_smoothing=0.0,
                      z_loss=0.0, softcap=0.0):
    """Fused CE backward -> grad_logits (T,V), scaled by per-row grad_out."""
    return _ext.cross_entropy_bwd(logits, targets, lse, grad_out, int(ignore_index),
                                  float(label_smoothing), float(z_loss), float(softcap))


def fused_linear_cross_entropy_ce(h, W, targets, row_weight=None, divisor=None,
                                  chunk_rows=2048, ignore_index=-100):
    """Liger-style chunked fused-linear-CE core: per-row CE of (h @ W.T) vs targets WITHOUT ever
    materializing the full (T,V) logits, with optional per-row loss weights (the §8 mixture w).

    h (T,K), W (V,K), targets (T,), row_weight (T,) or None; divisor defaults to the number of
    non-ignored rows. Returns (loss = sum(row_weight*ce)/divisor, dh (T,K), dW (V,K), ce (T,)) —
    grads w.r.t. h and W for that weighted mean; the grad w.r.t. row_weight is ce/divisor.
    """
    import torch as _t
    T = h.shape[0]
    if divisor is None:
        divisor = int((targets != ignore_index).sum().clamp(min=1).item())
    dh = _t.zeros_like(h)
    dW = _t.zeros_like(W)
    ce_all = _t.empty(T, dtype=_t.float32, device=h.device)
    total = _t.zeros((), dtype=_t.float32, device=h.device)
    for c0 in range(0, T, chunk_rows):
        c1 = min(c0 + chunk_rows, T)
        hc, tc = h[c0:c1], targets[c0:c1]
        logits = hc @ W.T                                      # (chunk, V) — the only big tensor
        loss_c, lse_c = cross_entropy_fwd(logits, tc, ignore_index)
        ce_all[c0:c1] = loss_c
        rw = row_weight[c0:c1].float() if row_weight is not None else None
        total = total + ((loss_c * rw).sum() if rw is not None else loss_c.sum())
        go = (rw / divisor) if rw is not None \
            else _t.full((c1 - c0,), 1.0 / divisor, device=h.device)
        g = cross_entropy_bwd(logits, tc, lse_c, go, ignore_index)
        dh[c0:c1] = (g @ W).to(h.dtype)
        dW += (g.T @ hc).to(W.dtype)
    return total / divisor, dh, dW, ce_all


def aum_silence_fwd(streams, alpha, xw, k_rot, halt_u, wpack, kappa, forced=-1, no_op=False,
                    halt_mode=0, delta=0.5):
    """Fused sequential global block, forward (§5-§9; roadmap step 4): the whole L-token silence
    recurrence in ONE kernel launch (one threadgroup per batch row). streams (B,L,2720) fp32 is
    the token-parallel precompute pack, wpack the flat weight pack (layouts:
    aum_ssm/ops/metal/silence_metal.py). no_op = the §14 stage-1 ablation (carry sigma^0).
    Returns (save (B,L,3151), j_star (B,L) int32, S_final,
    S_ckpt (B,ceil(L/64),8,64,64) — the segment-start states the backward replays from)."""
    return _ext.aum_silence_fwd(streams, alpha, xw, k_rot, halt_u, wpack, float(kappa),
                                int(forced), int(no_op), int(halt_mode), float(delta))


def aum_silence_bwd(streams, alpha, xw, k_rot, wpack, save, j_star, S_ckpt, dout, kappa,
                    forced=-1, no_op=False):
    """Fused sequential global block, backward: reverse march with segment S-replay. dout
    (B,L,2567) packs the incoming grads (d sigma_stack | d g_hat | d r_stack | d E | d pi | d w |
    d sigma_star). Returns (demit (B,L,5443) — per-token d-vectors from which the host forms all
    weight/stream grads as batched GEMMs — plus dalpha, dxw, dkrot)."""
    return _ext.aum_silence_bwd(streams, alpha, xw, k_rot, wpack, save, j_star, S_ckpt, dout,
                                float(kappa), int(forced), int(no_op))


def aum_decode(S, alpha, x, k_rot, q_rot):
    """Single-token U-phase decode step (§6): S <- alpha*S + x⊗k_rot ; out = S·q_rot.

    S (B,H,D,D) fp32 is updated in place; alpha (B,H), x/k_rot/q_rot (B,H,D) all fp32. MPS; D in
    {64,128}. x = rho*tau*v_hat (per p), k_rot/q_rot the rotated key/query (per n). Returns
    (out (B,H,D), S) — the D-skip and gated-RMSNorm are applied by the caller."""
    return _ext.aum_decode(S, alpha, x, k_rot, q_rot)
