# AUM-Ø Metal Kernel Plan (Apple Silicon, near-term) — rev. 2

## Context

AUM-Ø's hot kernel is the U-phase affine recurrence (§6). We target **two** GPU
backends over the same math, plus a portable oracle:

| Backend | Hardware | Status | Role |
|---|---|---|---|
| `reference` | CPU + **Apple MPS** | **done** (Phase 0) | pure-PyTorch oracle + portable fallback; **trains today on MPS** |
| `metal` | Apple Silicon (Metal) | **this plan — near-term** | local-dev acceleration via **ThunderMittens / `tk_torch`** |
| `triton` | **NVIDIA** | **deferred, NOT canceled** | ultimate production training path |

**rev. 2 — ThunderMittens matured a lot** since rev. 1 (it is being actively developed
in `/Users/eric/ThunderMittens`). Three findings rewrite the plan:

1. **`tk_torch` is a torch-native MPS path — no MLX, no DLPack.** It compiles the shared
   `.metal` kernels into `tk.metallib` with `xcrun metal`, JIT-builds a thin ObjC++
   extension (`torch.utils.cpp_extension.load`) that binds a torch MPS tensor's
   `MTLBuffer` and dispatches on **PyTorch's own MPS command stream**
   (`torch::mps::get_command_buffer/get_dispatch_queue`). Call site is just
   `import tk_torch; y = tk_torch.<kernel>(mps_tensors, scalars)` → torch MPS tensor.
   Requirements: PyTorch≥2.1 (MPS) + Xcode Metal toolchain only.
2. **The kernel bases already exist.** `mamba2` (chunked SSD), `lin_attn_decay`
   (RetNet/Lightning decay LA), `lin_attn_causal` (explicit running `D×D` state),
   `based`, `hedgehog`, `linear_attn`, plus `rotary`, `rope_kv`/`rope_kv_insert_norm`
   (fused K-RMSNorm+RoPE), `rms_norm`, `attn_bwd` (FA-2 backward scaffold).
3. **The substrate is ~90% ready** — `mma_AtB` (the rank-1 state update `S += Kᵀ·V`),
   the decay tile `exp(cumlog_i−cumlog_j)`, reductions, `make_causal`, rotary,
   `exp/tanh/silu` all present. Only genuine gaps: **`sigmoid` + `softplus` ops**, and
   (optional) an in-kernel float cumsum (the shipped kernels host-precompute `cumlog`).

**Cornerstone (verified):** the §6 reference runs **fwd+bwd on `mps`** at reference dims
(4 heads × 128). So the model + the §23 gate can run on the `reference` backend now; the
Metal kernel is a **speed track that does not block the gate**.

## Strategy: one math, pluggable backend, two independent tracks

The U phase dispatches on `AumConfig.kernel_backend ∈ {auto, reference, metal, triton}`
(default `auto` → `triton` on CUDA, `metal` on Apple if `tk_torch` importable, else
`reference`). Every backend is validated against `ssd_reference` (Phase 0).

- **Model/gate track** (Phases 1–5) runs on `reference` over MPS — no Metal dependency.
- **Metal track** (this doc) accelerates the U phase and the silence reads; it can proceed
  in parallel and be swapped in per-backend once grad-clean.

## Integration: `tk_torch` + `torch.autograd.Function` (replaces the old MLX/DLPack plan)

`tk_torch` is **forward-only** (no autograd anywhere in it — confirmed). So the `metal`
backend wraps each kernel call:

```python
# aum_ssm/ops/metal/unfold_metal.py  (metal backend)
import tk_torch                                    # from /Users/eric/ThunderMittens/.../kernels
class _UnfoldChunkMetal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, tau, lam, r, theta, z, D, dt_bias):
        # host prep (MPS): cumlog=cumsum(-lam*tau); cos/sin from phi; L2norm(k),L2norm(v);
        # X = (sigmoid(r)*tau)*v_hat ; C=rotate(q); B=rotate(k_hat)
        y = tk_torch.aum_unfold(C, B, X, cumlog, ...)   # the forked mamba2-core kernel
        ctx.save_for_backward(...); return y
    @staticmethod
    def backward(ctx, dy): ...                       # manual bwd (see kernel plan)
```

- Tensors stay on `mps`; `torch.mps.synchronize()` before any host read.
- bf16 activations, fp32 for `cumlog`/decay and mma accumulators (matches TM convention
  and the reference's fp32 state).
- The kernel itself lives in **ThunderMittens** (added via the `tk_torch` "add-a-kernel"
  recipe: `tk_launch.h` launch template → `torch_kernels.mm` `_mps` fn + `m.def` →
  `__init__.py` wrapper + `_METAL_SOURCES`). `aum_ssm` imports it through `tk_torch`; this
  is a documented cross-repo dependency (ThunderMittens on `PYTHONPATH` / pip-installed).

## Kernel plan (validated against `ssd_reference` at every step)

### Training (chunk-parallel) — fork `kernels/mamba2/mamba2.metal`
It already computes `O = ((C@Bᵀ) ⊙ L ⊙ causal) @ X`, `L[i,j]=exp(cumlog_i−cumlog_j)`.
Map: `C=R(φ)q`, `B=k_rot=R(φ)·L2norm(k)`, `X=ρτ·L2norm(v)`, `cumlog=cumsum(−λτ)`.
- **Decay:** *no kernel change* — feed `cumlog` from host.
- **Incremental fusion:**
  - *Stage 1 (fastest correct GPU path):* do rotary + L2-norm + ρτ + `cumlog` as PyTorch
    ops on MPS, call the (D=128-instantiated) mamba2 core for the SSD matmul+decay only.
    Near-zero new Metal code; validate vs `aum_unfold_chunk_ref`.
  - *Stage 2 (fused):* move rotary (single scalar phase/head), L2-norm, and the ρτ scale
    in-register (reuse `rope_kv_insert_norm`'s norm+rotate block), add `sigmoid`/`softplus`
    ops to `base_ops.metal`+`maps.metal`.
- **D=128:** all TM SSM kernels are `static_assert(D==64)`; add the `D=128` instantiation
  (rotary/rope_kv already ship D=128). 4 heads → grid `(N/8, 4, B)`.
- **Backward (the hard part):** none of the SSM kernels have one. Two-step:
  - *bridge:* Stage-1 forward + backward via autograd through the PyTorch prep and a
    reference-computed core-gradient (correct, not fast) — unblocks training on Metal.
  - *fused:* write the decay-weighted/rotated/gated backward (`dC,dB,dX,d cumlog→dλ`, and
    grads back through rotary φ, the L2-norms, and `ρ=sigmoid`) using the shipped
    `attn_bwd` (FA-2 dQ/dK/dV) family as the scaffold.

### Decode (single-token) — fork `kernels/lin_attn_causal/lin_attn_causal.metal`
It already holds an explicit `rt_fl<D,D>` state and does `KV += kᵀv` + `q@KV`. Deltas:
collapse to one token; persist `S` in device memory across steps (reuse `rope_kv`'s
read/modify/write-cache pattern); add scalar decay `S *= exp(−λτ)`; gated rank-1 update
`S += ρτ (v̂ ⊗ k_rot)`; readout `S·R(φ)q`; carry running `φ`; D=128. Inference-only → no
backward. Oracle: `aum_unfold_step_ref`.

### Silence reads — reuse the training kernel with a swapped query
`r_t^j = S_t R(φ_t) W_q^σ σ_t^j` is the same readout with `C = σ-query`; no new write.
Oracle: `aum_state_readout_ref`.

## Substrate additions to ThunderMittens (small)
1. `sigmoid` (`struct sigmoid` in `common/base_ops.metal` + wrappers in the 4 `maps.metal`,
   mirroring `silu`). 2. `softplus` (numerically-stable `log1p(exp(x))`). 3. *(optional)*
   a float prefix-scan for in-kernel `cumlog` (wrap `metal::simd_prefix_inclusive_sum`);
   otherwise keep host-precompute.

## Milestones (near-term; M0/M1 gate the rest, all MPS-validatable)

- **M0 — tk_torch smoke.** Put ThunderMittens on path; `import tk_torch`; run its
  `tests/test_mps.py`; confirm a `mamba2`/`lin_attn_decay` call round-trips a torch-MPS
  bf16 tensor. *(MPS already verified up.)*
- **M1 — run on Apple Silicon via reference.** U-phase module restructure (Appendix-A
  controller; make the Triton import lazy) + backend selector default `reference` →
  `AumLMHeadModel` trains fwd+bwd on `mps` end-to-end (silence off). "It trains on the Mac."
- **M2 — decode Metal kernel** (`lin_attn_causal` fork) vs `aum_unfold_step_ref`.
- **M3 — training Metal kernel, Stage 1** (mamba2 core + PyTorch prep) inside a
  `torch.autograd.Function`; fwd vs `aum_unfold_chunk_ref`, bwd via the reference bridge.
- **M4 — fuse (Stage 2)** rotary+norm+gate + `sigmoid`/`softplus` ops; **fused backward**
  via the `attn_bwd` scaffold, grad-checked vs autograd-through-reference; silence reads on
  the metal backend.
- **M5 — benchmark** metal vs reference-on-MPS; optimize (D=128 tiling, shared-memory
  state); optional in-kernel `cumlog` scan.

## Verification
- Correctness: every Metal kernel diffed against `ssd_reference` (CPU float64 oracle vs MPS
  bf16/fp32 with per-kernel tolerance, as `tk_torch`'s own tests do); backward grad-checked
  vs autograd-through-reference.
- Parity: decode == one-token chunk == reference step.
- Portability gate: `kernel_backend=reference` runs the whole model + Phases 2–5 on CPU and
  MPS with no Metal/Triton dependency.

## Risks
1. **Manual/fused backward** for the chunk kernel (hardest). Mitigation: reference is an
   exact gradient oracle; keep `reference` the default training backend until the fused
   backward is grad-clean; `attn_bwd` gives the scaffold.
2. **Cross-repo coupling** (aum_ssm ↔ ThunderMittens, actively developed). Mitigation:
   pin/import via `tk_torch`; the `metal` backend is optional (auto-falls back to reference).
3. **bf16 accuracy** of the affine scan on Apple GPUs. Mitigation: fp32 `cumlog`/state
   accumulation (reference parity), bf16 I/O only.
4. **D=128 / new activations** in TM. Mitigation: tiny, well-scoped additions with the
   shipping D=64 kernels + `silu`/`tanh` as templates; land as ThunderMittens PRs with tests.

## Relationship to the master plan
Refines Phase 1's kernel sub-task into the backend matrix. Phase 1.1 (module restructure)
is backend-agnostic and is **M1**. The old Triton in-kernel fusion is **deferred to NVIDIA
bring-up**. Phases 2–5 (silence, integration, training, gate) are unchanged and run on MPS
via `reference` (and `metal` once landed).
