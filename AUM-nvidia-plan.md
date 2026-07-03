# AUM-Ø NVIDIA Bring-up — Handoff (8× RTX 3090 node)

You are picking up a working, validated project at the point of porting it to NVIDIA. This
document is the complete brief: what exists, what is proven, what to build, in what order, and
the traps we already hit so you don't hit them twice. The spec is `AUM-Ø.md` (v6); architecture
notes in `AUM-design.md`; the Metal kernel campaign in `AUM-metal-plan.md` and git history.

## 0. State of this node (already done — do not redo)

- `~/AUM` and `~/ThunderMittens` cloned; `~/AUM/.venv`: torch 2.6.0+cu124, triton 3.2.0,
  accelerate/transformers/datasets/wandb/pytest/einops. 8× RTX 3090 (24GB), ~1TB RAM.
- **Corpus is ready**: `train/data/` holds 1,000,471,539 train tokens (244,255 × 4096-token
  windows, 11 shards) + 10.1M val — byte-identical to the corpus the Mac trained on
  (deterministic streaming). Loader verified. Do NOT re-run `prepare_data.py`.
- **Init checkpoint ready**: `train/checkpoints/aum-tiny-v6-init` (78,279,040 params, seed 0,
  matches the Appendix-A manifest).
- **CUDA smoke passed**: full model fwd+bwd on the *reference* (pure-PyTorch) path, loss 11.343
  — exactly matches the Mac. It is SLOW (14.3s @ B2 N1024): the global silence block is a
  per-token Python loop, launch-bound on CUDA too. That is the main thing you are here to fix.
- GPUs are at 350W; the owner wants `nvidia-smi -pl 200` (needs sudo — ask him, or a
  NOPASSWD sudoers rule for nvidia-smi).
- `train/muon.py::build_optimizer` already selects the **distributed** MuonWithAuxAdam when a
  multi-rank process group is initialized (DDP averages grads; Muon shards Newton-Schulz across
  ranks and all_gathers params). Single-device variant otherwise. Done — don't redo.

## 1. What this model is (60 seconds)

AUM-Ø-Tiny v6: d_model 512, 12 evidence layers (windowed GQA attention + a mamba2-SSD-family
"unfold" U-phase, 8 heads × 64), one **global silence block** on top — a TRUE sequential
recurrence over tokens (hypothesis register σ 128-d + the top layer's evidence state S
8×64×64), with loss-mixture halting (J_max=2 → 3 candidate outputs/token). Training = staged
schedule (§12) with 7 loss terms (§10). Vocab 49152 (SmolLM2), seq len 4096.

Two facts drive all kernel work:
1. **The U-phase IS mamba2 SSD** with operands pre-folded: C = R(φ)q, B = R(φ)k̂, X = ρτ·v̂,
   cumlog = cumsum(−λτ). R(φ) is a per-head multi-frequency rotation ladder.
2. **The silence block is thin but sequential**: ~25 small matvecs + 2 LayerNorms + 4 reads of
   S per token, with a 128-d carried register. Eager execution = ~50 kernel launches × 4096
   tokens × both directions. Fusing it was worth ~5× on Metal end to end.

## 2. Mission, in order

### A. Multi-GPU training loop (do this FIRST — the rig can train on the reference path
###    while you write kernels)

`train/train.py` currently refuses `num_processes > 1`. Changes needed:

1. **Trainer DDP safety** (`aum_ssm/training/trainer.py`): AumTrainer touches
   `self.model.backbone` and `self.model.lm_head`; under DDP those live on the wrapped module's
   `.module`. Keep the *wrapped* model for `self.model(...)` calls (grad sync), add
   `self.raw = accelerator.unwrap_model(model)` for attribute access. Same for
   `evaluate()`/`pred_val_r2` paths in train.py.
2. **train.py**: remove the `num_processes > 1` SystemExit; pass
   `DistributedDataParallelKwargs(find_unused_parameters=True)` to Accelerator — in stage 1 the
   halting/consistency heads get no gradient, and stages change at runtime, so the graph is not
   static. `tokens_per_step` must multiply by `accelerator.num_processes`.
3. **Rank-synchronized schedule**: stage advancement is gated on held-out pred-R² (§12).
   Val loaders shard per rank → each rank sees different batches → stages would diverge.
   `accelerator.gather()` the val loss and pred_r2, mean them, decide identically everywhere.
4. **Mixture-CE memory** (`aum_ssm/training/losses.py`): on CUDA the fused-Metal path is
   unavailable and the fallback materializes (B, L, J+1, V) logits — 2.4GB bf16 per 2
   sequences at 4096×49152×3. Chunk the fallback over rows (compute CE in ~4096-row slices,
   accumulate) until you build/wire a fused CE (see B4). This is a 15-line change; do it
   before the first 8-GPU run.
5. **bf16**: run with `--mixed-precision bf16`. Keep optimizer states, A_log, cumlog fp32
   (they already are).
6. Launch: `accelerate launch --num_processes 8 train/train.py ...` (or extend
   `train/launch.sh` to detect CUDA and exec accelerate launch). wandb is on by default when
   the package is installed; the owner is `wandb login`'d as ehartford, project `aum-ssm`.
7. Batch sizing on 24GB: start `--batch-size 2 --grad-accum 4` per rank (= 8×2×4×4096 = 262k
   tokens/step global; adjust warmup/eta accordingly or keep 65,536 by accum 1 — the §13 recipe
   says batch ≈ 0.5M tokens, so 8 ranks × B2 × accum 8 × 4096 ≈ 0.5M is actually the recipe).
   Measure memory before raising.
8. Validate: single-GPU short run (loss should start ~11.5 and fall exactly like the Mac run —
   wandb run `aum-tiny-v6-20260702-1927` is the reference curve); then 8-GPU smoke
   (~50 steps); check rank-0-saved checkpoints resume correctly.

### B. Triton kernel family (the Metal kernels are the SPEC — read them first)

Every kernel below already exists on Metal, validated to fp32 noise, with a pure-PyTorch oracle
in-repo. Your job is a port, not research. Order by measured impact:

**B1. Fused silence recurrence** — the big one (~90% of a step before fusing on Metal).
- Spec: `kernels/metal/src/aum_silence.metal`. The header comments document the whole design:
  the token-parallel PRECOMPUTED STREAMS (g/m/s/phase column blocks of every concat-Linear,
  P_G g, rotation cos/sin), the thin sequential core, the save-pack layout, the S checkpoints
  every 64 tokens, and the backward's reverse march with per-64-token segment replay.
- Oracle: `aum_ssm/ops/metal/silence_flat.py::flat_forward` — pure PyTorch, op-for-op equal to
  the production loop (validated ≤1e-6 vs `AumBackbone._global_segment`). Test YOUR kernel
  against this, then gradients against autograd through `_global_segment` itself.
- Host plumbing to REUSE verbatim: `aum_ssm/ops/metal/silence_metal.py` — pack builders
  (build_weight_pack / build_stream_pack define the exact buffer layouts), `unpack_save`, the
  `_SilenceFused` autograd.Function, and the entire backward GEMM assembly. Only the two
  `km.aum_silence_fwd/bwd` calls need Triton twins. Add a `silence_triton.py` mirroring it, or
  parameterize the backend in that file.
- Triton shape: one program (CTA) per batch row, `tl.dot`/vector ops over the 128/512-wide
  matvecs, sequential `for t in range(L)` — the same structure as fused-recurrent
  linear-attention kernels. Weights stay in global memory (7.6MB fp32; L2-resident). σ in
  registers/shared. S (8×64×64) in global, updated in place.
- Backward design rule (learned the hard way): the kernel EMITS per-token d-vectors (dz of
  every gate/projection, dq of every read, LN output grads) into a demit pack; ALL weight and
  stream gradients are then batched GEMMs over B·L rows on the host. Never accumulate weight
  grads in-kernel (outer product per matvec per token = bandwidth catastrophe), and in the host
  assembly use plain `mm` — `torch.einsum("tjo,tji->oi", ...)` materialized a (T,J,O,I)
  broadcast product (tens of GB) on MPS; don't trust it on any backend.
- Integration: `AumBackbone._fused_silence_ok` (aum_ssm/models/aum_lm.py) gates the fused
  route; extend it (device "cuda" + triton importable → the triton path). The reference loop
  stays as fallback for §14 ablations/baselines/decode. `--set silence_fused=False` pins the
  loop.
- Acceptance: fwd trajectories ≤1e-5 vs flat_forward; grads vs autograd through
  `_global_segment` at L=12, L=70 (crosses a segment boundary), and ablation="no_op"; full-
  model stage-1 step loss equal and all param grads matching. CAVEAT for the comparisons:
  under no_op + mixture loss, halting/consistency grads are ANALYTICALLY ZERO (equal dw across
  identical candidates cancels the cascade) — compare with an absolute noise floor (~1e-7),
  not pure relative error, or you will chase ghosts like we did.

**B2. Fused U-phase operand pipeline** (`aum_operands` + `aum_epilogue`).
- Spec: `kernels/metal/src/aum_unfold.metal`; host wiring `aum_ssm/ops/metal/unfold_metal.py`.
  One pass builds C/B/X from q/k/v/φ/ρτ (rotation ladder + two L2 norms + scaling + layout
  transpose); epilogue fuses D-skip + gated RMSNorm + transpose back. Worth 2.2–4× on the
  Metal forward path; the same fusion is unavailable to upstream mamba kernels.

**B3. Chunked SSD forward/backward.**
- Spec: `kernels/metal/src/mamba2.metal` — chunk L=64, kv→scan→out forward; backward =
  recompute Sex (kv→scan) + ONE reverse gradient chain (gstate→rscan) + row/col output kernels
  with the IN-KERNEL dcl split (rowsum(M) = r_intra + ⟨dC_inter, C_i⟩, colsum(M) = cc_intra +
  ⟨dX_inter, X_j⟩; host combines (r+ri)−(cc+ci)). bf16 scanned state (consumers take bf16
  anyway — free bandwidth). Tests: `tests/test_metal.py` has the NaN-safe quadratic oracle —
  note it masks the decay exponent with −inf BEFORE exp (exp(cl_i−cl_j) overflows in the upper
  triangle at N≳600; inf×0=NaN through masks/tril backward).
- BENCHMARK HONESTLY vs upstream: this repo is a mamba fork —
  `mamba_ssm/ops/triton/ssd_combined.py` (`mamba_chunk_scan_combined`) is Tri Dao's tuned
  implementation. Compare at exactly (B per-rank, H=8, N=4096, D=64) bf16, fwd and bwd,
  after adapting the cumlog↔dt·A parameterization. If upstream wins the bare core on 3090s,
  route to it INSIDE the fused operand pipeline and keep the fusion wins — measured routing
  with forced-route entry points and parity tests, exactly like the Metal auto-routes
  (`kernels/metal/__init__.py` docstrings show the pattern).

**B4. Fused mixture CE.** Metal spec: `kernels/metal/src/cross_entropy.metal` +
`fused_linear_cross_entropy_ce` + `_FusedMixtureCE` in `aum_ssm/training/losses.py` (the §8
mixture without materializing (B,L,3,V) logits — measured 2.2GB vs 25.6GB on Metal). On CUDA
the pragmatic move is Liger-kernel's fused_linear_cross_entropy under the same `_FusedMixtureCE`
interface (per-row weights = the halting w, divisor = B·L), or a direct Triton port.

**B5. Window attention** — nothing to do. `ground_attn._sliding_blocks` (block-local O(L·w),
checkpointed) is backend-agnostic and already the default for L > 2w. HISTORY: SDPA with a full
(L,L) window mask fell back to a path that saved the ENTIRE quadratic attention matrix
((B,8,4096,4096) fp32 ×2 ≈ 103GB at B=8) — that was the original memory killer. Don't
reintroduce mask-based SDPA. flash-attn's native window_size is a fine later swap if measured
faster.

### C. Launch the 1B run

When A lands (B1 makes it ~5× cheaper but is not a blocker): 8 ranks, B2 × accum 8 × 4096 ≈
0.5M tokens/step (the §13 recipe), bf16, wandb on. The reference loss trajectory for the first
~60 steps is wandb run `ehartford/aum-ssm/aum-tiny-v6-20260702-1927` (LM 10.84→9.84 by step 64
at 65,536 tok/step — scale expectations to your batch). Stage boundaries auto-compute; the §12
R² gate holds stage 1 until the prediction head beats trivial (expect R² < 0 for a long time —
that is correct, not a bug). Checkpoints every 1000 steps under `train/checkpoints/<run>/`.

## 3. Numerical traps we already hit (respect these)

1. Oracle decay masks: −inf into the exponent BEFORE exp (see B3).
2. μ enters the consistency functional DETACHED (§7); prec_G/prec_R weights still get grads.
3. Halting: p_J ≡ 1; w = cascade; j* via inverse-CDF on PRE-DRAWN uniforms (halt_u) so
   checkpoint/replay recompute reproduces the same sample. forced_depth → one-hot w with NO
   gradient into the p's (module uses F.one_hot; skip the halting backward entirely).
4. LayerNorm backward from saved (mean, rstd): dv = rstd·(dyw − mean(dyw) − x̂·mean(dyw·x̂)).
5. Analytically-zero grads under no_op (see B1 acceptance) — absolute noise floors in tests.
6. The S chain backward NEVER reconstructs S_{t−1} by dividing by α (α→0 blows up); replay
   forward per 64-token segment from the checkpoints the forward kernel wrote.
7. Muon: only 2D hidden matrices (min dim > 1); tied embedding/classifier and every
   scalar/vector/conv → AdamW. `partition_params` in train/muon.py implements the split.
8. Allocator-cache creep: on MPS the fix was `--empty-cache-every`. Watch
   `torch.cuda.memory_reserved()` growth on CUDA; if it creeps, the analogous
   `torch.cuda.empty_cache()` hook is already plumbed (currently MPS-gated — extend it).

## 4. Repo conventions

- Tests: `pytest tests/` — 87 pass + 1 skip on the Mac; Metal-only tests skip cleanly off-MPS.
  Every kernel lands with oracle-parity + grad tests in the suite, not just scratch scripts.
- Commits: imperative subject; body says WHAT and WHY with measured numbers ("X→Y ms, Nx");
  never claim a speedup you didn't measure. Push to `origin main`
  (git@github.com:QuixiAI/AUM.git). The Mac-side agent may also push — pull before you start.
- The division of labor between repos: model-specific kernels live HERE
  (kernels/… in-repo, self-contained builds); general-purpose kernels get ported back to
  ThunderMittens (~/ThunderMittens) after they prove out — see AUM-metal-plan.md and the
  NOTICE file for the pattern. For Triton the analogous home for generic pieces is
  `mamba_ssm/ops/triton/` or a new `kernels/triton/` — keep AUM-specific fusion in-repo.
- Config knobs are AumConfig fields; training overrides via `--set key=value` (validated).
  Kernel geometry is HARDCODED to the reference config (D=512, DS=128, DM=32, H=8, DH=64,
  J=2) with eligibility checks + reference fallback — keep that pattern.

## 5. Definition of done

1. `accelerate launch --num_processes 8` trains with distributed Muon, bf16, synchronized
   stages, and checkpoints that resume. Loss curve matches the Mac reference.
2. Triton silence kernel passes the B1 acceptance battery; step time drops accordingly
   (Metal datum: fused vs loop = 4.6s vs 22.1s at B2 N4096 fwd+loss+bwd; CUDA should do
   better).
3. SSD path benchmarked vs upstream at reference shapes with routing decided by measurement,
   parity-tested both routes.
4. Mixture CE never materializes (B,L,3,V).
5. The 1B run is training on 8 GPUs with wandb reporting, and `pytest tests/` is green.
