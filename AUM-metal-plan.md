# AUM-Ø Metal Backend (Apple Silicon) — rev. 3, as built

## Status

The Metal backend is **landed and self-contained**. Training (forward + fused backward) and
single-token decode of the U phase run on the Apple GPU via PyTorch MPS, validated against the
pure-PyTorch oracle at both head dims and end-to-end through the full model. rev. 2's
ThunderMittens cross-repo dependency is **gone**: the kernels build from source in this repo
([kernels/metal](kernels/metal)) with only Xcode's Metal toolchain (`xcrun metal`) and PyTorch.

| Backend | Hardware | Status | Role |
|---|---|---|---|
| `reference` | CPU + Apple MPS (+CUDA) | **done** | pure-PyTorch oracle + portable fallback; the correctness ground truth |
| `metal` | Apple Silicon | **done** (this doc) | local-dev acceleration; fwd + fused bwd + decode step on the GPU |
| `triton` | NVIDIA | deferred, NOT canceled | the production training path |

Selection: `AumConfig.kernel_backend ∈ {auto, reference, metal, triton}`. Every backend is
validated against `ssd_reference`; the `metal` backend falls back to the reference oracle for its
backward if `_FUSED_BWD` is disabled.

## The self-contained build (`kernels/metal/`)

```
kernels/metal/
  include/            vendored header-only MSL substrate (tile/type/op primitives; derived from
                      ThunderMittens, an Apple MSL port of ThunderKittens — see NOTICE)
  src/mamba2.metal        SSD forward, D in {64,128}
  src/mamba2_bwd.metal    SSD backward (dC, dB, dX), D in {64,128}
  src/aum_decode.metal    single-token U-phase step, D in {64,128}
  aum_metal.mm        ObjC++ dispatch onto torch's MPS command stream
                      (torch::mps::get_command_buffer / get_dispatch_queue)
  __init__.py         compiles src/*.metal -> aum.metallib via `xcrun metal` on import, JIT-builds
                      the extension (torch.utils.cpp_extension.load), exposes
                      mamba2 / mamba2_bwd / aum_decode
```

Independence was verified adversarially: the D=128 tests pass while the ThunderMittens working
tree provides only D=64 — nothing is read from outside this repo at build or run time.

## The design that made this cheap (unchanged by v6 — by construction)

The U-phase chunk math **is** the Mamba-2 SSD numerator:

$$Y=\big((C B^\top)\odot e^{\,\mathrm{cl}_i-\mathrm{cl}_j}\odot\text{causal}\big)X,\qquad
C=R(\phi)q,\ B=R(\phi)\hat k,\ X=\rho\tau\hat v,\ \mathrm{cl}=\mathrm{cumsum}(-\lambda\tau)$$

All AUM-specific transforms — the §4 **multi-frequency rotation ladder**, k/v L2-norms, the
write gate, the dynamics — are folded into the kernel **operands** on the host (PyTorch ops on
MPS). Consequence: the v5.3→v6 rotation change (single phase → geometric ladder, φ unwrapped)
required **zero kernel edits**; only the host preamble changed. bf16 I/O, fp32 `cumlog` and
accumulators, per the reference's fp32-state convention.

### Forward — `mamba2` (as planned in rev. 2, plus D=128)
One simdgroup per (batch, head, query-chunk); decay tile from `add_row`/`sub_col`/`exp` over the
host-precomputed `cumlog`; `make_causal` on the diagonal chunk. Grid `(N/8, H, B)`.

### Backward — `mamba2_bwd` (the rev.-2 "hard part", now landed)
Forked from the FA-2 `attn_bwd` structure, atomics-free via the i/j ownership split:
`mamba2_bwd_i` (fix query chunk, loop j ≤ i) → `dC`; `mamba2_bwd_j` (fix key chunk, loop i ≥ j)
→ `dB, dX`. No softmax ⇒ no L/delta passes. The decay gradient never touches the kernel: the
host identity

$$d\,\mathrm{cl}_k=\langle dY_k,Y_k\rangle-\langle dX_k,X_k\rangle$$

(rowsum − colsum of $dM\odot M$) gives `dcumlog` from tensors already in hand. Wrapped in a
`torch.autograd.Function` (`aum_ssm/ops/metal/unfold_metal.py`); grads match the fp32 reference
within bf16 tolerance at both head dims; `_ssd_core_ref` retained as the oracle/fallback.

### Decode — `aum_decode` (replaces the rev.-2 `lin_attn_causal` fork plan)
A single token makes mma's 8×8 tiles waste 7/8 of their capacity, so the plan's tile-kernel fork
was dropped for a plain kernel: one threadgroup per (batch, head), **one thread per state row p**
— each row's decay + rank-1 write + readout is independent, so no cross-thread reduction and no
padding:

$$S[p,n] \leftarrow \alpha\,S[p,n] + x[p]\,k_{\text{rot}}[n],\qquad
\text{out}[p]=\textstyle\sum_n S[p,n]\,q_{\text{rot}}[n]$$

fp32 state (it is a recurrence), q/k staged in threadgroup memory. Wired as the `metal` decode
step (`aum_unfold_step_metal`), a drop-in for `aum_unfold_step_ref`; decode ≡ full forward holds
end-to-end through the real model, silence on and off.

## What v6 changed for this backend

1. **Rotation ladder is host-side** — kernels untouched (above).
2. **The chunked silence-read plan is obsolete.** rev. 2 planned to serve the silence reads with
   the training kernel under a swapped query. v6's global block is a *sequential* token
   recurrence (C7): the silent-read queries depend on σ_{t−1}, so no chunk-parallel readout
   exists. The reads are served from the sequentially-stepped S inside the backbone loop
   (per-token einsum on MPS); training memory is handled by exact-gradient segment checkpointing
   (`config.silence_segment`), which is implemented and bit-exact against the unsegmented loop.
3. **`sigmoid`/`softplus` substrate ops were never needed** — the gate/dynamics stayed in the
   host preamble.

## Milestones (rev.-2 plan → outcome)

| Milestone | Outcome |
|---|---|
| M0 tk_torch smoke | done (rev.-2 era); superseded by the vendored build |
| M1 reference on MPS | done — the whole model + gate machinery run backend-free |
| M2 decode kernel | done as `aum_decode` (design changed: plain per-row kernel, not a tile fork) |
| M3 chunk fwd + bridge bwd | done (`mamba2` core + PyTorch preamble in an autograd.Function) |
| M4 fused backward | done (`mamba2_bwd` + the dcumlog host identity) |
| M5 benchmark/fuse | **open** — see below |

## Remaining work (the open perf track)

1. **Fully-fused bespoke U-phase kernel** (fwd+bwd): fold the rotation ladder, L2-norms, ρτ
   gate, scan, and gated-RMSNorm into one kernel, eliminating the host preamble's extra
   global-memory passes. Tracked as QuixiAI/AUM issue #1. Note the ladder means per-block
   frequencies in-register — the rev.-2 "single scalar phase per head" fusion sketch no longer
   applies.
2. **Fused sequential global-block kernel** (C7): one dispatch running the
   σ→ĝ→e→μ→σ⁰→σ recurrence over the sequence with S resident on-chip
   (working state O(d_σ + d + H·d_h²) ≈ 66K floats per batch row at Tiny scale). Memory is
   already solved by checkpointing; this buys wall-clock.
3. **Benchmarks** (M5): metal vs reference-on-MPS at reference shapes; then D=128 tiling and
   shared-memory-state experiments per the perf handbook discipline.
4. **Triton/NVIDIA bring-up** (out of scope here): same operand-folding design; the SISO kernels
   in `ops/triton/unfold/` need the write gate, k/v norms, unwrapped φ, and the host-side ladder.

## Verification (standing, all in `tests/test_metal.py` + the suite)

- Kernel vs oracle: forward, backward (vs autograd through the PyTorch SSD core), and decode
  step, at D=64 and D=128, with bf16-appropriate tolerances.
- End-to-end: full model reference ≡ metal (fwd + grads finite); decode ≡ full forward through
  the real model on the metal backend, silence on and off.
- Portability gate: `kernel_backend=reference` runs everything (model, training harness, gate)
  on CPU and MPS with no Metal/Triton dependency.

## Risks (rev.-2 list, closed out)

1. ~~Manual/fused backward~~ — landed; the reference remains the gradient oracle and the
   in-tree fallback.
2. ~~Cross-repo coupling with ThunderMittens~~ — eliminated by vendoring the substrate
   (attribution in `kernels/metal/NOTICE`).
3. bf16 accuracy — handled as planned: fp32 `cumlog`/state, bf16 I/O; tolerances encode it.
4. ~~D=128 / new activations~~ — D=128 instantiations landed; no new substrate ops were needed.
