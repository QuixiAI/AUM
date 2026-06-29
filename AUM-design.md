# AUM-Ø Implementation Design — Forking the Mamba codebase

Implementation plan for forking [`state-spaces/mamba`](https://github.com/state-spaces/mamba)
into **`aum_ssm`**, the reference implementation of **AUM-Ø v5.3** (see `AUM-Ø.md`).

This document records the repo survey, the keep/remove decisions, and the
high-level map of what each surviving file becomes. It is the build plan, not
the architecture spec — the architecture lives in `AUM-Ø.md`.

## Decisions (settled)

1. **Kernels: SISO + Triton-only.** Keep only the Mamba-3 SISO Triton kernels.
   Delete the Mamba-1 CUDA build (`csrc/`), all TileLang/CuTe, and all MIMO.
   Matches the v5.3 reference (`mimo_rank=1`). Drops the
   `tilelang` / `apache-tvm-ffi` / `quack-kernels` dependencies and the CUDA
   compile step.
2. **Reference modules: keep `ssd_minimal` only.** Delete Mamba-1 and the
   Mamba-2 SSD kernels; keep `ssd_minimal.py` as a readable pure-PyTorch
   reference to validate the U-phase kernel against.
3. **Package identity: rename to `aum_ssm`.** Rewrite imports and metadata to
   the new architecture.

---

## Top-line finding

AUM-Ø decomposes into three tiers, unevenly covered by the repo:

| Tier | What it is | Repo coverage |
|---|---|---|
| **Evidence core** (12× A→U→M→MLP) | the token-clock recurrent stack | **~70% reusable** — essentially **Mamba-3** + GQA + a small precision adapter |
| **Global silence block** (`model.silence.*`) | hypothesis register, predictive grounding, pressure, halting | **0% — net-new**, no analog in the repo |
| **Training harness** (§17–20) | 4-stage schedule, counterfactual-benefit rollout, the 7-term loss | **0% — net-new**; the repo has no trainer at all |

**Key discovery: the U phase is Mamba-3, almost line-for-line.**
- `modules/mamba3.py:27` `heavy_tail_activation` is *exactly* the dissolution
  `f(x)` from §6.
- `ops/triton/mamba3/angle_dt.py:94-107` computes `tanh(angle)·π · dt`, cumsum,
  `mod 2π` — *literally* the resonance-phase recurrence
  `φ_t = (φ_{t-1} + π·tanh(θ_t)·τ_t) mod 2π`.
- Rotary `R(φ)` on q/k, gated RMSNorm readout, data-dependent decay, the SSD
  chunk-scan — all already present.

So the evidence core is "Mamba-3 SISO + a write-gate + key/value normalization,"
not a from-scratch kernel. The **affine invariant** committed in §0.2 is exactly
what keeps the chunk-scan kernel applicable.

**Cost note:** the **silence block needs no custom kernel.** With `J_max=2` and
frozen `S_t`, it is a tiny fixed unroll of 128-dim GEMMs, vectorizable across
`(B,T)` in plain PyTorch. The only heavy kernel is the U-phase scan.

---

## Spec → file correspondence

| Spec component | Closest existing file | Action |
|---|---|---|
| A — bounded GQA grounding | `modules/mha.py` | adapt: local/windowed causal + QK-norm |
| U — resonant affine recurrence | `modules/mamba3.py` + `ops/triton/mamba3/*` (SISO) | fork as the U core |
| φ resonance recurrence | `ops/triton/mamba3/angle_dt.py` | reuse ~as-is |
| heavy-tail `f(x)` | `modules/mamba3.py:27` | reuse verbatim |
| gated readout / B,C norm | `ops/triton/layernorm_gated.py` | keep |
| M — precision adapter | *(none — small low-rank module)* | new |
| MLP (SwiGLU) | `modules/mlp.py` | keep, minor reconcile |
| Block / residual stream | `modules/block.py` | restructure to host A/U/M/MLP |
| Backbone + LM head + init/save | `models/mixer_seq_simple.py` | adapt |
| **Silence block (all of it)** | *(none)* | **new `modules/silence.py`** |
| Generation / inference cache | `utils/generation.py` | adapt for σ/φ state + silence loop |
| Paired determinism (§17) | `utils/determinism.py` | reuse — helps the counterfactual rollout |
| **Trainer (§17–20)** | *(none)* | **net-new** |

---

## Target fork layout: `aum_ssm/`

```
aum_ssm/
  __init__.py                    ← exports AumLMHeadModel, AumConfig
  models/
    config_aum.py                ← was config_mamba.py  (extend: d_sigma, J_max, λ's, β, δ, p_explore)
    aum_lm.py                    ← was mixer_seq_simple.py  (12 evidence layers + silence block + LM head)
  modules/
    evidence_layer.py            ← was block.py        (restructure to host A→U→M→MLP)
    ground_attn.py               ← was mha.py          (A: local/windowed GQA + QK-norm)
    unfold.py                    ← was mamba3.py        (U: resonant affine + write-gate + k/v norm)
    modulate.py                  ← NEW                  (M: low-rank precision adapter)
    silence.py                   ← NEW                  (the entire global silence block)
    mlp.py                       ← kept (SwiGLU, minor reconcile to 3-matrix form)
    ssd_reference.py             ← was ssd_minimal.py   (read-only crutch for kernel validation)
  ops/triton/
    unfold/                      ← was mamba3/ (SISO only)
      siso_fwd.py  siso_bwd.py  siso_combined.py  siso_step.py
      angle_dt.py  grouped_head_reduction.py  utils.py
    layer_norm.py  layernorm_gated.py  k_activations.py  softplus.py
  utils/
    generation.py  hf.py  torch.py  determinism.py
  training/                      ← NEW (none of this exists in the repo)
    trainer.py  losses.py  schedule.py  counterfactual.py
    tasks/                       ← synthetic §22 generators
tests/                           ← adapt test_mamba3_siso + test_layernorm_gated; add AUM tests
setup.py  pyproject.toml  README.md  LICENSE  ...
```

---

## Keep / Adapt / Remove

### KEEP & ADAPT (the spine)
- `models/mixer_seq_simple.py`, `models/config_mamba.py`
- `modules/block.py`, `modules/mlp.py`, `modules/mamba3.py`, `modules/mha.py`
- `ops/triton/mamba3/` — **SISO only**: `mamba3_siso_fwd/bwd/combined/step.py`,
  `angle_dt.py`, `grouped_head_reduction.py`, `utils.py`
- `ops/triton/layer_norm.py`, `ops/triton/layernorm_gated.py`,
  `ops/triton/k_activations.py`, `ops/triton/softplus.py`
- `utils/generation.py`, `utils/hf.py`, `utils/torch.py`, `utils/determinism.py`
- `setup.py` (gut the CUDAExtension), `pyproject.toml`, packaging files,
  `LICENSE` / `README` (rewrite)

### KEEP AS REFERENCE (read-only crutches)
- `modules/ssd_minimal.py` — readable pure-PyTorch SSD; template for a non-kernel
  AUM-Ø reference forward to test kernels against.
- `tests/ops/triton/test_mamba3_siso.py`, `tests/ops/triton/test_layernorm_gated.py`
  — adapt into AUM kernel tests.

### REMOVE
```
csrc/                                       # entire Mamba-1 CUDA tree → also gut CUDAExtension in setup.py
mamba_ssm/modules/mamba_simple.py
mamba_ssm/modules/mamba2.py
mamba_ssm/modules/mamba2_simple.py
mamba_ssm/ops/selective_scan_interface.py
mamba_ssm/ops/triton/selective_state_update.py
mamba_ssm/ops/triton/ssd_bmm.py  ssd_chunk_scan.py  ssd_chunk_state.py  ssd_combined.py  ssd_state_passing.py
mamba_ssm/ops/tilelang/                     # all MIMO TileLang
mamba_ssm/ops/cute/                         # all CuTe
mamba_ssm/ops/triton/mamba3/mamba3_mimo_*.py   (fwd_varlen, bwd, bwd_varlen, rotary_step, mimo_utils)
mamba_ssm/distributed/                      # defer scaling (re-add later if needed)
evals/  benchmarks/                         # defer (first eval is §22 synthetic, not lm-harness)
tests/ops/test_selective_scan.py  tests/ops/tilelang/  tests/ops/cute/  tests/modules/test_mamba3_varlen.py
```

**Dependency fallout (the payoff):** dropping CUDA + TileLang lets
`pyproject.toml` lose `tilelang==0.1.8`, `apache-tvm-ffi`, `quack-kernels`, and
`setup.py` becomes a pure-Python package (no `--no-build-isolation` dance).
Runtime deps shrink to `torch, triton, einops, transformers` + optional
`causal-conv1d`.

---

## What each KEPT file becomes (work map)

| File | Becomes | Effort |
|---|---|---|
| `mixer_seq_simple.py` → `aum_lm.py` | Backbone loop changes from "N identical blocks" to "12 evidence layers (residual stream of `h^M`) → **one silence block** → norm → tied LM head." Preserve `_init_weights` incl. the `_no_reinit` hooks (matters for `dt_bias`/`A_log` init), `from_pretrained`/`save_pretrained`. | Med |
| `block.py` → `evidence_layer.py` | Restructure the single-mixer wrapper into the A→U→M ordering, where `h^{M,ℓ}=h^A+h^U+Δh` feeds the residual stream (§8). Top layer additionally emits `g_t`. | Med |
| `mha.py` → `ground_attn.py` | Add **QK-norm** (`q_norm/k_norm[64]`) and **bounded/windowed causal** attention (SDPA mask or flash local window). Strip the generation rotary-cache complexity not needed. | Med |
| `mamba3.py` → `unfold.py` | Fork the SISO path. Reuse `heavy_tail_activation` verbatim (= `f(x)`). **Add** write-gate `ρ_t=σ(r_t)` and L2-normalized `k̂,v̂` (B_norm/C_norm get close). Drop MIMO/tilelang/cute imports. | **High** |
| `ops/triton/mamba3/*` (SISO) → `ops/triton/unfold/` | `angle_dt.py` reused ~as-is (it *is* the φ recurrence). `siso_fwd/bwd` need the write-gate term threaded through fwd + grad. One extraction job: pull the rotary-step helper from `mamba3_mimo_rotary_step.py` into `siso_step.py` so the deleted MIMO file isn't referenced. | **High** |
| `layernorm_gated.py`, `layer_norm.py`, `k_activations.py`, `softplus.py` | Keep as-is (gated RMSNorm readout, fused add-norm, Triton helpers). | None |
| `mlp.py` | Keep; optionally split fused `fc1` into `gate_proj/up_proj` to match Appendix-A param names. | Low |
| `ssd_minimal.py` → `ssd_reference.py` | Golden pure-PyTorch reference: extend to the affine `S_t=α_t S_{t-1}+ρ_tτ_t(v̂⊗k̂)` form to unit-test the U kernel. | Low |
| `utils/generation.py` | Adapt the inference-cache loop to carry `(φ_t, S_t, σ_t)` and run the per-token silence unroll at decode. | Med |
| `utils/determinism.py` | Reuse directly for §17 **paired-determinism** counterfactual benefit (shared dropout masks / precision path). | None |

---

## Net-new (no file to fork — the bulk of the project)

- **`modules/silence.py`** — predict (`q^pred`, ĝ_t, e_t) · error-fed precision
  (μ_t, ẽ_t) · register init + the `J_max=2` revision loop · consistency `E_t` ·
  pressure `π_t` · soft halting `w_j` · condition output. All plain PyTorch, no
  kernel (frozen `S_t`, tiny unroll). Maps 1:1 to `model.silence.*` in Appendix A.
- **`modules/modulate.py`** — the M-phase low-rank `U·diag(μ)·V` precision adapter.
- **`training/`** — the 4-stage schedule (§20) with the pred-loss gate, the
  frozen-downstream benefit rollout (§17), the 7-term loss (§18),
  forced-exploration (§19), and the σ-decode/σ-intervention diagnostics (§21).
  **None of this exists** in the repo.
- **`training/tasks/`** — the four §22 synthetic generators with controlled
  evidence-age (branch reversal, binding swap, delayed correction, flat null).

---

## Known gotchas to handle during the port

- **U-kernel modifications.** The U phase adds two things Mamba-3 SISO lacks: the
  per-step sigmoid **write-gate** `ρ_t=σ(r_t)` and **L2-normalized** `k̂,v̂`.
  `B_norm`/`C_norm` approximate the normalization; the write-gate is a real (small)
  change to `siso_fwd/bwd`.
- **Rotary-step extraction.** `mamba3.py` imports `apply_rotary_qk_inference_fwd`
  from the (to-be-deleted) `mamba3_mimo_rotary_step.py`. Extract the SISO rotary
  step into `siso_step.py` before deleting the MIMO files.
- **Init hooks.** Preserve the `_no_reinit` / `_no_weight_decay` markers on
  `dt_bias` and `A_log` — they encode the targeted Δ/A initialization the spec
  relies on; a generic trainer that re-zeros biases would break them.

---

## Scoping read

The repo hands over a working, well-tested **evidence core** cheaply (especially
the U-phase kernel), but the **silence block** and the **entire training/eval
harness** are greenfield — and the harness is where the falsifiable claims in
§22–23 actually get tested.

## Suggested first execution step

On a fresh branch: do the `git mv`/renames and deletions, rewrite imports, gut
the CUDA build from `setup.py`, trim `pyproject.toml` deps, and stub out
`silence.py` / `modulate.py` / `training/` with signatures from Appendix A so the
package imports and compiles as a skeleton.
