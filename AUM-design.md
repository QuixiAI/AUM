# AUM-Ø Implementation Design — the Mamba fork, as built

Design record for the fork of [`state-spaces/mamba`](https://github.com/state-spaces/mamba)
into **`aum_ssm`**, the reference implementation of **AUM-Ø v6** (see `AUM-Ø.md`).

This document records the repo survey, the keep/remove decisions, what each surviving file
became, and the design consequences discovered during the build. It is the build record, not the
architecture spec — the architecture lives in `AUM-Ø.md`, whose Appendix A is verified
tensor-for-tensor against the built checkpoint. **Status: executed.** The v6 reference model
(78,255,136 params) is implemented, tested (decode ≡ forward, kernel ≡ oracle, the §14/§15 gate
machinery), and trains fwd+bwd on CPU and Apple MPS.

## Decisions (settled, executed)

1. **Kernels: SISO scaffolding + pluggable backends.** The Mamba-1 CUDA build (`csrc/`), all
   TileLang/CuTe, and all MIMO were deleted. The U phase dispatches on
   `AumConfig.kernel_backend ∈ {auto, reference, metal, triton}`: `reference` is the
   pure-PyTorch oracle (CPU/MPS/CUDA), `metal` is the self-contained Apple-Silicon backend
   (see `AUM-metal-plan.md`), `triton` is deferred to the NVIDIA production bring-up.
2. **Reference module: `ssd_minimal` → `ssd_reference.py`.** Extended into the full AUM oracle:
   the affine recurrence in serial (`aum_unfold_step_ref`) and chunk-parallel
   (`aum_unfold_chunk_ref`) forms, the swapped-query readout (`aum_state_readout_ref`), the
   dynamics (`aum_dynamics`), and the §4 rotation ladder (`ladder_freqs`, `_rotate_ladder`).
   Step ≡ chunk to 1e-9 in float64 is a standing test.
3. **Package identity: `aum_ssm`.** Imports, metadata, and packaging rewritten; pure-Python
   package (the CUDA extension and the `tilelang`/`apache-tvm-ffi`/`quack-kernels` dependencies
   are gone).

---

## Top-line finding (held up in practice)

AUM-Ø decomposes into three tiers, unevenly covered by the original repo:

| Tier | What it is | Repo coverage (as experienced) |
|---|---|---|
| **Evidence core** (12× A→U→M→MLP) | the token-clock recurrent stack | **~70% reused** — Mamba scaffolding + GQA + a small precision adapter |
| **Global silence block** (`model.silence.*`) | hypothesis register, predictive grounding, pressure, halting | **net-new**, no analog in the repo |
| **Training harness** (§10–§12, §14–§16) | 4-stage schedule, counterfactual-benefit labels, the 7-term loss, the gate | **net-new**; the repo has no trainer at all |

**The key identity that carried the whole kernel strategy:** the U-phase readout

$$S_t\,\tilde q_t \;=\; \big((C B^\top)\odot e^{\,\mathrm{cumlog}_i-\mathrm{cumlog}_j}\odot\text{causal}\big)X
\quad\text{with}\quad C=R(\phi)q,\; B=R(\phi)\hat k,\; X=\rho\tau\hat v,\; \mathrm{cumlog}=\mathrm{cumsum}(-\lambda\tau)$$

is *exactly* the Mamba-2 SSD numerator. Every AUM-specific transform (rotation ladder, L2
normalization, write gate, dynamics) folds into the kernel's **operands**, computed host-side.
This is why the same SSD core serves the reference, Metal, and (eventually) Triton backends
unchanged — and why v6's multi-frequency ladder (§4) required **zero kernel changes**.

**Mamba-3 correspondence — a v5.3-era note, corrected for v6.** Mamba-3's
`heavy_tail_activation` is the dissolution `f(x)` (reused as `heavy_tail`), and its angle/dt
machinery matches the phase-velocity recurrence. But two v6 deltas break line-for-line reuse:
v6's φ is an **unbounded accumulated position, never wrapped mod 2π** (the wrap is harmless under
a single frequency, aliasing-relevant under the ladder), and the rotation applies **B = d_h/2
geometric frequencies per head**, not one. The Triton `angle_dt.py` kernel therefore needs those
two edits at NVIDIA bring-up time; the reference and Metal paths already implement v6 semantics.

**Cost note, revised by v6.** The v5.3 claim "the silence block needs no kernel — a tiny unroll
vectorizable across (B,T)" died with the two-pass carry. v6's global block is a **true sequential
recurrence over tokens** (C7) that steps the top layer's S alongside σ to serve its reads. It is
implemented as a per-token loop under **exact-gradient segment checkpointing**
(`config.silence_segment`, boundary carries only, pre-drawn `halt_u` uniforms so the Categorical
j* reproduces under recompute) — memory-solved; the fused sequential kernel remains the
wall-clock optimization (see `AUM-metal-plan.md`).

---

## As-built layout

```
aum_ssm/
  __init__.py                  exports AumLMHeadModel, AumConfig (lazy, PEP 562 — importing the
                               package never requires Triton/Metal)
  models/
    config_aum.py              defaults ARE the Tiny v6 reference (§13)
    aum_lm.py                  backbone: 12 evidence layers -> the sequential global recurrence
                               (segment-checkpointed) -> tied LM head; decode is one more step of
                               the same recurrence
  modules/
    evidence_layer.py          A -> U -> M -> MLP around the prenorm residual stream; stashes the
                               per-layer mu^l for the §10 scoped l1
    ground_attn.py             A: windowed causal GQA (8/2 x 64, w=256) + QK-norm + KV-cache decode
    unfold.py                  U: controller/projections/conv, backend dispatch, the rope_freqs
                               ladder buffer, prefill state capture, decode step, and the
                               write-pack the global recurrence consumes
    modulate.py                M: low-rank U.diag(mu).V precision adapter
    silence.py                 the entire global block (§5-§9): predict, precision, register,
                               consistency (detached-mu), pressure (no H_t), loss-mixture halting
                               (o_stack + one carried sigma^{j*}), J(pi) policy, ablations,
                               Top-GRU baseline with the learned pool-query head
    mlp.py                     SwiGLU (fused fc1 = [gate; up] — Appendix A matches this layout)
    ssd_reference.py           the pure-PyTorch oracle (see Decisions #2)
    norm.py                    RMSNorm (pure PyTorch; CPU/MPS/CUDA)
  ops/
    metal/unfold_metal.py      the metal backend: autograd.Function over the self-contained
                               kernels (fwd mamba2, fused bwd mamba2_bwd + the dcumlog host
                               identity, decode step aum_decode)
    triton/                    layer_norm, layernorm_gated, k_activations, softplus + unfold/
                               (SISO kernels, DEFERRED to NVIDIA bring-up; imports are lazy)
  training/
    losses.py                  the §10 objective incl. the §8 LM mixture; per-layer-only mu l1
    counterfactual.py          fixed-K labels (§11) + the on-policy gauge
    schedule.py                stages, the scale-free R^2 pressure gate, p_explore floor
    trainer.py                 stage-aware step tying all of it together
    tasks/synthetic.py         §14 families with held-out generators (disjoint value ranges)
    diagnostics.py             corr(pi,b), Delta-sigma quartet, sigma-decode, sigma-relevance,
                               phase-velocity stats
    gate.py                    the §14/§15 harness: variants, controls, the phase-distance
                               falsifier, evidence-survival probe, the MVP table checks
  utils/                       generation (prefill+decode driver, transformers-optional), hf,
                               determinism, torch helpers
  distributed/                 kept (tensor-parallel helpers for the NVIDIA bring-up)
kernels/metal/                 self-contained Metal build — no external kernel repo (see
                               AUM-metal-plan.md)
train/                         init.py (the ~78M checkpoint), tokenizer.py (SmolLM2, 49152),
                               prepare_data.py (uint16 shards), muon.py (§13 optimizer recipe)
tests/                         71 tests: oracle equivalences, rotation-ladder anti-aliasing,
                               silence semantics, decode==forward (CPU+MPS, silence on/off),
                               metal==reference (both head dims, fwd/bwd/decode), checkpointing
                               exactness, trainer/gate machinery, optimizer partition
```

---

## What each kept file became (work map, closed out)

| Origin | Became | Notes |
|---|---|---|
| `mixer_seq_simple.py` | `models/aum_lm.py` | backbone loop + the v6 sequential global recurrence; `_init_weights` with `_no_reinit`/`_no_weight_decay` hooks preserved (they carry the `dt_bias`/`A_log` priors) |
| `block.py` | `modules/evidence_layer.py` | A→U→M ordering, `h^M` into the residual stream, top layer exposes the silence ctx |
| `mha.py` | `modules/ground_attn.py` | QK-norm + windowed causal + KV-cache decode |
| `mamba3.py` | `modules/unfold.py` | Appendix-A layout; `heavy_tail` reused; write gate + k/v L2-norm added; backend dispatch; `rope_freqs` buffer |
| `ssd_minimal.py` | `modules/ssd_reference.py` | grew into the full oracle |
| `ops/triton/mamba3/` (SISO) | `ops/triton/unfold/` | parked for NVIDIA bring-up (needs the v6 ladder + unwrapped φ edits) |
| `utils/generation.py` | kept | decode driver carries (φ, S, σ) via the inference cache; CUDA-graph path CUDA-only; transformers optional |
| `utils/determinism.py` | kept | serves §11 paired determinism |
| `mlp.py` | kept | fused `fc1`/`fc2`; Appendix A was regenerated to match the build rather than splitting the weights |

Everything on the v5.3 REMOVE list was removed; `distributed/` was retained (deferred, not
deleted). Dependency fallout as predicted: runtime deps are `torch, einops` (+ optional
`transformers` for generation conveniences, `triton` only on NVIDIA).

## Net-new (was the bulk of the project; all landed)

`modules/silence.py`, `modules/modulate.py`, the whole `training/` tree (schedule, losses,
counterfactual, trainer, tasks, diagnostics, gate), the decode paths, the Metal backend and the
self-contained `kernels/metal` build, and the `train/` pipeline front-end (init → tokenizer →
data prep → Muon).

## Gotchas that materialized (and their resolutions)

- **The two-pass σ-carry was a dead end.** v5.3's TBPTT-1 two-pass enabled chunk-parallel
  silence reads but blended trained σ-dynamics with an approximation the spec never committed
  to; v6 replaced it with the true sequential recurrence + segment checkpointing. Decode and
  training now run literally the same per-token step — the strongest parity invariant in the
  test suite.
- **The predictive read's phase.** The v5.3-era read closure rotated the §5 predictive query at
  φ_t; the spec says φ_{t−1}. Caught and fixed when the sequential loop made the read explicit.
- **Lazy imports everywhere.** Eager package imports pulled Triton on machines without it;
  `__getattr__` (PEP 562) at the package root and lazy backend imports keep `reference` usable
  with zero GPU deps.
- **Manifest drift.** Appendix A had accumulated seven divergences from the build (fused MLP,
  silence module paths, LN biases, `conv1d.bias`, `condition_norm`, `pressure_out` as a bare
  vector, approximate counts). Resolved by regenerating the appendix from the real state dict
  and adding a programmatic 331-entry equality check.

## What remains

- `train/train.py` — the pretraining loop (memmap shard loader, Muon + warmup/cosine, the §12
  staged schedule, checkpoint/resume). Everything it needs already exists.
- Triton/NVIDIA bring-up (`ops/triton/unfold/`): thread the write gate + k/v norms through
  `siso_fwd/bwd`, switch `angle_dt` to unwrapped φ, apply the rotation ladder host-side exactly
  as the Metal backend does.
- The two remaining Metal kernels (fused bespoke U-phase, fused sequential global block) — see
  `AUM-metal-plan.md`.
