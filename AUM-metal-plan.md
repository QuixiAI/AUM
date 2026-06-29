# AUM-Ø Metal Kernel Plan (Apple Silicon, near-term)

## Context

AUM-Ø's hot kernel is the U-phase affine recurrence (§6). We now target **two** GPU
backends over the same math, plus a portable oracle:

| Backend | Hardware | Status | Role |
|---|---|---|---|
| `reference` | CPU + **Apple MPS** | **done** (Phase 0) | pure-PyTorch oracle + portable fallback; runs/ trains today |
| `metal` | Apple Silicon (Metal/MLX) | **this plan — near-term** | local-dev acceleration on the Mac |
| `triton` | **NVIDIA** | **deferred, NOT canceled** | the ultimate production training path |

**Decision (user):** Triton is *deferred*, not dropped — final training runs on NVIDIA,
so the existing `ops/triton/unfold/` SISO kernels stay and the ρ/k-v-norm fusion (the
old Phase 1.2) is rescheduled to the NVIDIA bring-up, not removed. Near-term development
happens on Apple Silicon, so we add a Metal backend now.

**Cornerstone fact (verified):** the §6 reference (`aum_unfold_chunk_ref`) runs
forward **and backward** on `mps` at reference dims (B=2, L=16, 4 heads, headdim 128),
fp32. So AUM-Ø can be brought up and trained on the Mac via the `reference` backend
immediately; the Metal kernel only has to *beat the reference on MPS*, with the
reference as its correctness oracle.

## Strategy: one math, pluggable backend

The U phase calls a dispatcher selecting the backend from config
(`AumConfig.kernel_backend ∈ {auto, reference, metal, triton}`, default `auto`):
- `auto` → `triton` on CUDA, `metal` on Apple (if built), else `reference`.
- `reference` always available (the oracle); used for tests and any unsupported shape.

This keeps the model identical across hardware and lets every backend be validated
against `ssd_reference` (Phase 0). The silence reads (§4/§10) are the *same* readout with
a swapped query, so they reuse whichever backend is active (`aum_state_readout_*`).

## Why MLX `mx.fast.metal_kernel`, not the full ThunderMittens build (initially)

ThunderMittens is an early MSL port of ThunderKittens (3 kernels: `add_rt`,
`matmul_custom`, `attn_fwd`; no SSM/linear-attention kernel; **missing primitives**:
sequence cumsum/segsum, outer-product accumulate into a state tile, rotary pair-rotation,
L2-norm reduction) and needs MLX-from-source + Xcode to build. Standing up a fused SSD
kernel on its primitives is itself a multi-step sub-project.

`mx.fast.metal_kernel` lets us write MSL **inline in Python**, JIT-compiled and callable on
MLX arrays — the fastest path to a *correct* Metal kernel with no separate build. We start
there; we graduate the kernel onto structured ThunderMittens primitives later, once it is
correct and we want the tiled/optimized version (and TM's primitives have matured).

**Reference templates** (under `/Users/eric/ThunderMittens/.reference/`):
- Decode recurrence (Metal-native): `vllm-metal/.../kernels_v2/gdn_recurrent_decode.metal`
  — per-head scalar decay, k·state / q·state via `simd_sum`, rank-1 state update.
- Chunk-parallel training: `ThunderKittens/kernels/linear_attention/linear_attention.cu`
  — intra-chunk (Q·Kᵀ masked + decay) vs inter-chunk (state·=decay; state+=Kᵀ·V) split.
- Rotary + norm: `mlx/.../kernels/rope.metal`, `rms_norm.metal`.

## Integration architecture: Hybrid (torch backbone + MLX Metal U-phase)

Keep the model in **PyTorch/MPS** (ecosystem, the existing harness, and the eventual
NVIDIA/Triton path); replace only the U-phase op with an MLX Metal kernel:
- **Bridge:** torch-MPS ↔ MLX via **DLPack** (`mx.from_dlpack` / `torch.from_dlpack`),
  zero-copy on the Metal device.
- **Autograd:** DLPack carries data only, so wrap the MLX kernel in a
  `torch.autograd.Function` with a **manual backward**. The backward formulas are
  grad-checked against autograd-through-`ssd_reference` (which is exact).
- **Not** a full MLX port of the model (MLX's nn/autograd is less mature; the harness,
  HF interop, and the NVIDIA path are torch). Revisit only if the bridge proves a
  bottleneck.

Requires: `pip install mlx` into the venv (torch MPS already present).

## Kernels to build (each validated against `ssd_reference`)

1. **Decode (single-token), `aum_unfold_step_metal`** — inference-only, no grad first
   (simplest). SIMD-per-head: load state `S (Dv×Dqk)`; `S*=exp(-λτ)`; `S += ρτ (v̂⊗k̂_rot)`;
   `r = S·q_rot`; gated-RMSNorm readout. Oracle: `aum_unfold_step_ref`. Template:
   `gdn_recurrent_decode.metal`.
2. **Chunk-parallel (training), `aum_unfold_chunk_metal`** — fwd via simdgroup-matrix
   intra/inter-chunk SSD scan; **manual backward** in the `torch.autograd.Function`.
   Oracle: `aum_unfold_chunk_ref` (fwd) and autograd-through-reference (grads). Template:
   `linear_attention.cu` structure on Apple `simdgroup_matrix`.
3. **Silence readout** — the swapped-query readout reuses kernel 2 with `Q=σ-query`,
   no new write. Oracle: `aum_state_readout_ref`.

Rotary (`R(φ)`, single data-dependent phase per head), k/v L2-norm, and the ρ write-gate
are applied **inside** the Metal kernel (or as cheap MLX pre-passes), mirroring the
reference; `R(φ)` is orthogonal so norm/rotary order is free.

## Milestones (near-term, all CPU/MPS-validatable on the Mac)

- **M0 — bridge smoke.** `pip install mlx`; verify `torch.backends.mps`, a trivial
  `mx.fast.metal_kernel`, and a DLPack round-trip torch-MPS↔MLX. *(verified: MPS up.)*
- **M1 — run on Apple Silicon via reference.** Land the U-phase **module restructure**
  (controller/in_proj_qkv/in_proj_dyn etc., Appendix A — pure PyTorch) + the backend
  selector defaulting to `reference`. Result: `AumLMHeadModel` forward+backward on `mps`
  end-to-end (silence off). This is "it trains on the Mac." No kernel yet.
- **M2 — decode Metal kernel** via `mx.fast.metal_kernel`, validated vs
  `aum_unfold_step_ref` on MPS (fp32, then fp16/bf16 tolerance).
- **M3 — chunk training Metal kernel** + `torch.autograd.Function` (fwd vs
  `aum_unfold_chunk_ref`; bwd grad-checked vs autograd-through-reference).
- **M4 — silence readout backend** + wire the silence block (Phase 2) to the active
  backend; one-token == decode parity.
- **M5 — benchmark** Metal vs reference-on-MPS; optimize; optionally restructure onto
  ThunderMittens primitives (and add the missing TM primitives: seq-cumsum/segsum,
  outer-product accumulate, rotary, L2-norm) for the tiled version.

## Verification

- **Correctness:** every Metal kernel output diffed against `ssd_reference` — CPU float64
  oracle vs MPS float32/bf16 with appropriate tolerance; backward via
  `torch.autograd.gradcheck`-style comparison against autograd-through-reference.
- **Parity:** decode kernel == one-token chunk == reference step.
- **Portability gate:** with `kernel_backend=reference`, the full model + all Phase 2–5
  tests run on CPU and MPS with no Metal/Triton dependency.

## Risks

1. **MLX↔torch DLPack on MPS** maturity (nightly torch + MLX version skew). Mitigation:
   M0 smoke gates everything; fall back to `reference` on any bridge failure.
2. **Manual backward** for the chunk kernel — error-prone. Mitigation: the reference gives
   an exact gradient oracle; keep `reference` as the default training backend until the
   Metal backward is grad-clean.
3. **ThunderMittens immaturity** if/when we move off `mx.fast.metal_kernel`. Mitigation:
   inline kernels first; structured TM port is M5, optional.
4. **fp16/bf16 accuracy** of the affine scan on Apple GPUs. Mitigation: fp32 state
   accumulation (as the reference does), reduced-precision I/O only.

## Relationship to the master plan

This refines the master plan's Phase 1 kernel sub-task into a backend matrix. Phase 1.1
(module restructure) is backend-agnostic and is M1 here. The old Phase 1.2 (in-kernel
Triton fusion) is **deferred to NVIDIA bring-up**. Phases 2–5 (silence, integration,
training, gate) are unchanged and run on MPS via the `reference`/`metal` backends.
