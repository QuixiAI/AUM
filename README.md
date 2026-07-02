# AUM-Ø

**Attentive Unfolding Modulation with Silence** — an affine resonant evidence core with a
benefit-gated global hypothesis register.

> **AUM-Ø v6** (pronounced *Aum-nought*) is a recurrent sequence architecture built on one
> principle — *continuation arises from temporary configuration* — and one structural commitment:
> *separate evidence from interpretation, and spend extra computation only where revising the
> interpretation pays.*
>
> Spec: [AUM-Ø.md](AUM-Ø.md) (the source of truth — architecture, training recipe, pre-registered
> evaluation, and a tensor manifest verified against the built checkpoint).

The architecture maintains two kinds of state on two clocks. An **evidence state** $S_t$ — a
phase-addressed associative memory updated once per token by an affine recurrence — records *what
has been observed*. A **hypothesis register** $\sigma_t$ — a small nonlinear state revised zero or
more times per token by an inner "silence" loop — holds *how the evidence is currently
interpreted*. A learned **integration pressure** $\pi_t$, trained against measured counterfactual
benefit, decides when revision is worth the compute:

$$A \rightarrow U \rightarrow M \rightarrow \varnothing:\quad
\text{observe} \rightarrow \text{accumulate} \rightarrow \text{weigh} \rightarrow \text{revise when it pays}$$

This repository is a fork of [state-spaces/mamba](https://github.com/state-spaces/mamba): the
evidence core is a gated linear-attention / selective-SSM recurrence in the Mamba-2 family, and
the U-phase chunk kernel is mathematically the Mamba-2 SSD numerator with AUM's rotation,
normalization, and gating folded into its operands.

## Status

The **AUM-Ø-Tiny v6 reference model (78,255,136 params)** is fully implemented and validated:

- **Evidence core** (12 layers of A→U→M→MLP): bounded local GQA grounding, the resonant affine
  evidence recurrence with the multi-frequency rotation ladder (§4), error-free precision, SwiGLU.
- **Global silence block** (§5–§9, ~1.77M params): hypothesis-conditioned predictive grounding,
  error-fed precision, the revision loop with loss-mixture halting (per-candidate outputs, one
  carried register — no state blending), integration pressure, and the fixed-depth-K
  counterfactual benefit pipeline.
- **True sequential global recurrence** (§2/C7) with exact-gradient segment checkpointing, so
  seq-8192 training never materializes the per-token state chain.
- **Decode**: single-token generation is one more step of the same recurrence —
  decode ≡ full forward is a test invariant, silence on and off.
- **Training harness**: staged schedule (§12) with the scale-free R² pressure gate, the seven-term
  objective (§10), synthetic task families with held-out generators, the full baseline/control/
  ablation gate (§14/§15), and the registered diagnostics (§16).
- **Apple-Silicon Metal backend**: self-contained kernels (forward, fused backward, decode step)
  in [kernels/metal](kernels/metal) — no external kernel repo needed, only Xcode's Metal
  toolchain. Training fwd+bwd runs fully on the GPU via PyTorch MPS.
- **Deferred**: Triton/NVIDIA in-kernel fusion (the production training path); the fused
  sequential kernel for the global block (wall-clock optimization — memory is already solved).

## Layout

```
AUM-Ø.md              the v6 specification (source of truth)
AUM-design.md         fork/implementation design notes
AUM-metal-plan.md     Metal backend plan
aum_ssm/
  models/             AumConfig (defaults = the Tiny v6 reference) + AumLMHeadModel
  modules/            evidence_layer, ground_attn (A), unfold (U), modulate (M),
                      silence (the global block), ssd_reference (the pure-PyTorch oracle)
  ops/metal/          the Metal U-phase backend (dispatch onto kernels/metal)
  training/           losses, schedule, trainer, counterfactual benefit, synthetic tasks,
                      diagnostics, and the §14/§15 gate harness
  utils/              generation (prefill + decode driver)
kernels/metal/        self-contained Metal build: MSL substrate + mamba2 (SSD fwd),
                      mamba2_bwd (SSD bwd), aum_decode (single-token step)
train/
  init.py             materialize the randomly-initialized Tiny v6 checkpoint (~78M)
  tokenizer.py        SmolLM2 tokenizer (49152-vocab BPE — matches the spec exactly) + verify
  prepare_data.py     tokenize a corpus into packed uint16 shards + manifest
  muon.py             Muon optimizer (vendored) + the AUM parameter partition (§13 recipe)
tests/                the full suite (decode parity, kernel-vs-oracle, gate machinery, ...)
```

## Requirements

- PyTorch ≥ 2.x. The reference backend is pure PyTorch and runs on **CPU, Apple MPS, or CUDA** —
  no Triton, no CUDA toolkit needed.
- For the Metal backend: an Apple-Silicon Mac with Xcode's Metal toolchain (`xcrun metal`).
  Kernels JIT-compile on first import.
- For data prep: `pip install transformers datasets numpy`.

## Quickstart

```bash
# 1. materialize the reference checkpoint (78,255,136 params; validates the Appendix-A manifest)
python train/init.py

# 2. verify the tokenizer against the model config (SmolLM2, vocab 49152 — an exact match)
python train/tokenizer.py --config train/checkpoints/aum-tiny-v6-init/config.json

# 3. tokenize a corpus into packed uint16 shards
python train/prepare_data.py --source HuggingFaceFW/fineweb-edu --streaming \
    --out-dir train/data/fineweb-edu --val-fraction 0.01 \
    --config train/checkpoints/aum-tiny-v6-init/config.json

# run the test suite
pytest tests/
```

### Model usage

```python
import torch
from aum_ssm.models.config_aum import AumConfig
from aum_ssm.models.aum_lm import AumLMHeadModel

cfg = AumConfig(silence_enabled=True)          # defaults ARE the Tiny v6 reference (~78M)
model = AumLMHeadModel(cfg)

ids = torch.randint(0, cfg.vocab_size, (1, 128))
logits = model(ids).logits                      # training/prefill forward

out = model.generate(input_ids=ids[:, :32], max_length=64, cg=False)   # recurrent decode

# training aux (per-candidate outputs, halting weights, pressure, consistency, ...)
result, aux = model(ids, return_aux=True)
```

`silence_enabled=False` gives the parameter-matched **evidence-core baseline** (~76.5M);
`baseline="top_gru"` the Top-GRU adapter; `ablation=...` at forward time selects the §14
mechanism-isolating controls (`no_op`, `no_read`, `phase_scrambled`, `random`).

### Optimizer (§13 recipe)

```python
from train.muon import build_optimizer
opt = build_optimizer(model)   # Muon (lr 0.02 spectral, wd 0.1) on 2D hidden matrices,
                               # AdamW (6e-4) on the tied embedding/classifier + scalars
```

### U-phase backends

`AumConfig(kernel_backend=...)`:

| backend | where | notes |
|---|---|---|
| `reference` (auto) | CPU / MPS / CUDA | pure PyTorch, the correctness oracle |
| `metal` | Apple MPS | self-contained kernels; fwd + fused bwd + decode step on the GPU |
| `triton` | NVIDIA | deferred to the production training bring-up |

## Reference configuration (AUM-Ø-Tiny v6)

| Field | Value |
|---|---|
| d_model / evidence layers | 512 / 12 (+1 global block) |
| Vocab (tied) | 49 152 (SmolLM2 BPE) |
| A: heads / kv / head-dim / window | 8 / 2 / 64 / 256 |
| U: heads / head-dim / rotation ladder | 4 / 128 / B=64, ω ∈ [10⁻³, 1] geometric |
| Register d_σ / precision k_μ / J_max | 128 / 32 / 2 |
| Params: total / silence / ablated core | 78,255,136 / 1,769,408 / 76,485,728 |

## Provenance

- Forked from [state-spaces/mamba](https://github.com/state-spaces/mamba) (Gu & Dao) — the
  backbone scaffolding, and the SSD formulation the U phase builds on. Original license retained
  ([LICENSE](LICENSE)).
- The Metal substrate in `kernels/metal/include` derives from ThunderMittens (an Apple MSL port of
  [ThunderKittens](https://github.com/HazyResearch/ThunderKittens)); see
  [kernels/metal/NOTICE](kernels/metal/NOTICE).
- `train/muon.py` vendors [Muon](https://github.com/KellerJordan/Muon) (Keller Jordan, MIT).
