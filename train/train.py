#!/usr/bin/env python
"""AUM-Ø training driver (v6 §10-§13): packed shards -> Accelerate -> Muon -> staged schedule.

Pieces it ties together:
  - DATA   : the uint16 shards written by train/prepare_data.py, read as NON-OVERLAPPING
             ``seq_len``-token windows (default 4096). Documents are EOS-separated in the flat
             stream, so a book-length document is automatically split into consecutive 4k chunks
             — no truncation, no fake document boundaries injected inside it.
  - MODEL  : AumLMHeadModel from an init/resume checkpoint directory (config.json +
             pytorch_model.bin, the train/init.py format). Attention is SDPA —
             GroundAttention calls F.scaled_dot_product_attention (torch picks the
             flash/mem-efficient kernel per device); nothing to configure here.
  - OPTIM  : train/muon.py ``build_optimizer`` — Muon (lr 0.02 spectral, momentum 0.95, wd 0.1)
             for the 2D hidden matrices, AdamW (6e-4, betas (0.9,0.95), no wd) for the tied
             embedding/classifier and all scalars; shared 1500-step warmup + cosine to 10% (§13).
  - LOOP   : HF Accelerate (device placement, gradient accumulation, optional mixed precision)
             around AumTrainer's stage-aware step; the §12 schedule advances on the token-budget
             split 60/20/15/5 with the load-bearing R^2 gate before stage 2 (L_pressure stays off
             until the prediction head beats the trivial predictor on held-out data), the
             lambda_C 0 -> 5e-3 ramp inside stage 3, and the p_explore 0.2 -> 0.02 anneal.

    # the default run: ONE PASS over the prepared corpus (train/data manifest sets the budget;
    # --epochs N for more passes, --total-tokens to pin an explicit budget)
    python train/train.py

    # hyperparameters: optimizer knobs are flags; loss weights / model constants are AumConfig
    # fields, overridable with --set (validated against the config schema)
    python train/train.py --muon-lr 0.01 --adamw-lr 3e-4 --warmup-steps 500 \
        --set lambda_pred=0.7 --set lambda_pressure=0.5

    # a laptop smoke run: 30 optimizer steps at short context
    python train/train.py --total-tokens 500000 --seq-len 512 --batch-size 1 --grad-accum 4 \
        --warmup-steps 5 --eval-every 10 --save-every 20 --run-name smoke

    # evidence-core baseline (silence stack off; plain LM loss)
    python train/train.py --no-silence --run-name evidence-core

    # resume
    python train/train.py --resume train/checkpoints/<run>/step-000200

Progress is a tqdm bar over optimizer steps (loss/stage/lr/tok-s live in the postfix); periodic
loss-part lines, eval results, and stage transitions print above it and append to
<run>/metrics.jsonl. W&B reporting is ON by default whenever the wandb package is installed
(train/* per log interval, val/* per eval, lr, tokens, stage; --wandb-project to name the
project, WANDB_MODE=offline works, --no-wandb to disable; an init failure degrades to a
warning, never kills the run).

Multi-GPU: `accelerate launch --num_processes 8 train/train.py ...` (train/launch.sh does this
automatically on a CUDA node). DDP averages gradients (find_unused_parameters=True — stage-gated
heads have no grad path in early stages and the graph changes at stage boundaries); train/muon.py
selects the DISTRIBUTED MuonWithAuxAdam under a multi-rank process group (Newton-Schulz sharded
across ranks, params all_gathered). The §12 stage schedule is rank-synchronized: val loss and the
R^2 gate statistic are all-reduced means, so every rank advances stages identically. Checkpoints:
rank 0 saves the model + its trainer state; every other rank saves trainer_state_rank{r}.pt
(Muon momentum is sharded by rank) — resume with the SAME num_processes to restore momentum
exactly (a missing shard falls back to rank 0's state: fresh momentum, still correct).
"""

import argparse
import ast
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aum_ssm.models.aum_lm import AumLMHeadModel                      # noqa: E402
from aum_ssm.models.config_aum import AumConfig                       # noqa: E402
from aum_ssm.training import losses as L                              # noqa: E402
from aum_ssm.training.trainer import AumTrainer                       # noqa: E402
from aum_ssm.training.schedule import (Stage, ScheduleConfig, p_explore_at,   # noqa: E402
                                       pressure_gate_clear)
from train.muon import build_optimizer                                # noqa: E402


# --------------------------------------------------------------------------- data
class PackedWindows(Dataset):
    """Non-overlapping seq_len-token windows over the flat uint16 shard stream.

    This IS the 4k chunking: window i of shard s is tokens [i*seq_len, (i+1)*seq_len) — a long
    document spans as many consecutive windows as it needs, short documents pack together with
    their EOS separators. The per-shard tail (< seq_len tokens) is dropped.
    """

    def __init__(self, data_dir, split, seq_len):
        with open(os.path.join(data_dir, "manifest.json")) as f:
            manifest = json.load(f)
        if split not in manifest["splits"]:
            raise KeyError(split)
        self.seq_len = seq_len
        self.paths, self.index = [], []            # index[i] = (shard_idx, window_idx)
        for s in manifest["splits"][split]["shards"]:
            path = os.path.join(data_dir, s["name"])
            n_win = s["n_tokens"] // seq_len
            self.index.extend((len(self.paths), w) for w in range(n_win))
            self.paths.append(path)
        self.vocab_size = manifest["vocab_size"]
        self._maps = {}                            # lazy per-shard memmaps (worker-safe)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        shard, w = self.index[i]
        mm = self._maps.get(shard)
        if mm is None:
            mm = self._maps[shard] = np.memmap(self.paths[shard], dtype=np.uint16, mode="r")
        a = np.asarray(mm[w * self.seq_len:(w + 1) * self.seq_len], dtype=np.int64)
        return torch.from_numpy(a)


def make_loader(dataset, batch_size, seed, shuffle=True):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g,
                      drop_last=True, num_workers=0)


def cycle(loader):
    while True:
        yield from loader                          # a fresh pass reshuffles via the sampler


# --------------------------------------------------------------------------- schedule pieces
def lr_scale(step, warmup, total, floor=0.10):
    """§13: linear warmup then cosine to `floor` of the peak. Multiplies each group's base lr."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def stage_boundaries(total_steps, fractions):
    """Cumulative §12/§13 token-budget boundaries in optimizer steps: [end1, end2, end3]."""
    ends, acc = [], 0.0
    for f in fractions[:-1]:
        acc += f
        ends.append(int(acc * total_steps))
    return ends


# --------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, raw, cfg, val_iter, n_batches):
    """Held-out LM loss (the §8 mixture when the silence stack is on, plain CE otherwise).
    `model` may be DDP-wrapped (forward goes through it); `raw` is the unwrapped module for
    attribute access. Per-rank shard mean — the caller reduces across ranks."""
    was_training = model.training
    model.eval()
    raw.backbone.silence_enabled = cfg.silence_enabled
    losses = []
    for _ in range(n_batches):
        x = next(val_iter)
        if cfg.silence_enabled:
            _, aux = model(x, return_aux=True)
            losses.append(float(L.lm_mixture_loss(aux.o_stack[:, :-1], aux.w[:, :-1],
                                                  raw.lm_head, x[:, 1:])))
        else:
            losses.append(float(L.lm_loss(model(x).logits[:, :-1], x[:, 1:])))
    if was_training:
        model.train()
    return sum(losses) / max(1, len(losses))


# --------------------------------------------------------------------------- checkpointing
def load_model(ckpt_dir, overrides):
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        d = json.load(f)
    d.update(overrides)
    model = AumLMHeadModel(AumConfig(**d))
    state = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"),
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def save_checkpoint(accelerator, model, optimizer, run_dir, step, stage, tokens_seen):
    """Called on ALL ranks: rank 0 saves the model + trainer_state.pt; every other rank saves
    trainer_state_rank{r}.pt — the distributed Muon shards momentum by rank, so exact resume
    needs each rank's optimizer state (Adam states are replicated; the duplication is the
    price of a simple exact resume)."""
    path = os.path.join(run_dir, f"step-{step:06d}")
    if accelerator.is_main_process:
        os.makedirs(path, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(path)
    accelerator.wait_for_everyone()                  # dir exists before non-zero ranks write
    state = {"optimizer": optimizer.state_dict(), "step": step, "stage": int(stage),
             "tokens_seen": tokens_seen}
    name = "trainer_state.pt" if accelerator.is_main_process \
        else f"trainer_state_rank{accelerator.process_index}.pt"
    torch.save(state, os.path.join(path, name))
    if accelerator.is_main_process:
        latest = os.path.join(run_dir, "latest")
        if os.path.islink(latest):
            os.remove(latest)
        os.symlink(os.path.basename(path), latest)
    accelerator.wait_for_everyone()
    return path


# --------------------------------------------------------------------------- main
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Train AUM-Ø (Accelerate + Muon + staged §12 schedule).")
    ap.add_argument("--init", default=os.path.join(here, "checkpoints", "aum-tiny-v6-init"),
                    help="init checkpoint dir (train/init.py output)")
    ap.add_argument("--resume", default=None, help="checkpoint dir (step-NNNNNN) to resume from")
    ap.add_argument("--data-dir", default=os.path.join(here, "data"))
    ap.add_argument("--run-name", default="aum-tiny-v6")
    ap.add_argument("--out-dir", default=os.path.join(here, "checkpoints"))
    # §13 recipe knobs
    ap.add_argument("--total-tokens", type=float, default=None,
                    help="token budget (default: --epochs passes over the prepared corpus)")
    ap.add_argument("--epochs", type=float, default=1.0,
                    help="passes over the corpus when --total-tokens is not given")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=2, help="micro-batch (sequences)")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=1500)
    ap.add_argument("--muon-lr", type=float, default=0.02,
                    help="Muon peak lr, spectral-norm units (§13: 0.02)")
    ap.add_argument("--muon-momentum", type=float, default=0.95)
    ap.add_argument("--adamw-lr", type=float, default=6e-4,
                    help="AdamW peak lr for embedding/scalars (§13: 6e-4)")
    ap.add_argument("--weight-decay", type=float, default=0.1, help="Muon matrices only")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any AumConfig field (loss weights etc.), e.g. "
                         "--set lambda_pred=0.7 --set lambda_pressure=0.5 --set kappa=0.2; "
                         "repeatable, applied last")
    ap.add_argument("--lambda-compute-max", type=float, default=5e-3, help="stage-3 ramp target")
    ap.add_argument("--eta-r2", type=float, default=None,
                    help="override the §12 R^2 pressure gate (default 0.15; e.g. -1e9 to force "
                         "stage progression in smoke tests)")
    ap.add_argument("--no-silence", action="store_true",
                    help="evidence-core baseline: silence stack off, plain LM loss, no stages")
    ap.add_argument("--kernel-backend", default=None,
                    help="override config (auto|reference|metal|triton)")
    ap.add_argument("--mixed-precision", default="no", choices=["no", "bf16", "fp16"])
    # cadence
    ap.add_argument("--eval-every", type=int, default=250, help="optimizer steps between evals")
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--r2-len", type=int, default=512, help="context length for the R^2 gate probe")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=1,
                    help="optimizer steps between log lines / wandb points (steps are minutes "
                         "at 4k context on MPS — log every one)")
    ap.add_argument("--empty-cache-every", type=int, default=None,
                    help="release the allocator cache every N optimizer steps (0 = never; "
                         "default 10 on MPS, 0 on CUDA). On MPS the cache otherwise grows "
                         "monotonically (~85GB/step peak at 8x4096), gets paged out, and step "
                         "time decays; on CUDA watch torch.cuda.memory_reserved() and enable "
                         "this only if it creeps")
    ap.add_argument("--no-tqdm", action="store_true", help="plain-print progress (logs/CI)")
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None,
                    help="report to Weights & Biases (default: ON when the wandb package is "
                         "installed; --no-wandb to disable)")
    ap.add_argument("--wandb-project", default="aum-ssm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.empty_cache_every is None:                  # MPS needs it; CUDA usually not
        args.empty_cache_every = 10 if torch.backends.mps.is_available() else 0

    if args.wandb is None:                              # default: report whenever wandb exists
        try:
            import wandb  # noqa: F401
            args.wandb = True
        except ImportError:
            args.wandb = False

    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    # find_unused_parameters: stage-gated heads (halting/consistency/pressure) have no grad
    # path in stage 1, and the active graph changes at stage boundaries — the DDP graph is
    # not static.
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum,
                              mixed_precision=args.mixed_precision,
                              log_with="wandb" if args.wandb else None,
                              kwargs_handlers=[DistributedDataParallelKwargs(
                                  find_unused_parameters=True)])
    set_seed(args.seed)

    # ---- model
    overrides = {"seq_len": args.seq_len, "silence_enabled": not args.no_silence}
    if args.kernel_backend:
        overrides["kernel_backend"] = args.kernel_backend
    fields = {f.name for f in dataclasses.fields(AumConfig)}
    for kv in args.set:                                  # --set KEY=VALUE, applied last
        key, eq, val = kv.partition("=")
        if not eq:
            raise SystemExit(f"--set expects KEY=VALUE, got {kv!r}")
        if key not in fields:
            raise SystemExit(f"--set: {key!r} is not an AumConfig field ({sorted(fields)})")
        try:
            overrides[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            overrides[key] = val                         # plain string (e.g. kernel_backend)
    ckpt = os.path.join(args.resume) if args.resume else args.init
    model = load_model(ckpt, overrides)
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())

    # ---- data (the 4k chunking lives here — see PackedWindows)
    train_ds = PackedWindows(args.data_dir, "train", args.seq_len)
    try:
        val_ds = PackedWindows(args.data_dir, "val", args.seq_len)
    except KeyError:
        val_ds = None
    if val_ds is None or len(val_ds) == 0:
        accelerator.print("WARNING: no val split — eval + the §12 R^2 gate use TRAIN windows")
        val_ds = train_ds
    train_loader = make_loader(train_ds, args.batch_size, args.seed)
    val_loader = make_loader(val_ds, args.batch_size, args.seed + 1)

    # ---- steps / schedule
    tokens_per_step = (args.batch_size * args.grad_accum * args.seq_len
                       * accelerator.num_processes)
    total_tokens = args.total_tokens if args.total_tokens is not None \
        else len(train_ds) * args.seq_len * args.epochs   # default: --epochs corpus passes
    total_steps = max(1, math.ceil(total_tokens / tokens_per_step))
    schedule = ScheduleConfig() if args.eta_r2 is None else ScheduleConfig(eta_r2=args.eta_r2)
    ends = stage_boundaries(total_steps, schedule.stage_fractions)   # [s1_end, s2_end, s3_end]

    # ---- optimizer (§13 split) + trainer
    optimizer = build_optimizer(model, muon_lr=args.muon_lr, embed_lr=args.adamw_lr,
                                scalar_lr=args.adamw_lr, momentum=args.muon_momentum,
                                weight_decay=args.weight_decay)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    raw_model = accelerator.unwrap_model(model)
    # Forward through the WRAPPED model (DDP grad sync); attribute access through raw.
    trainer = AumTrainer(model, optimizer, cfg, schedule, accelerator=accelerator,
                         raw=raw_model)

    start_step, tokens_seen = 0, 0
    if args.resume:
        # Muon momentum is sharded by rank (train/muon.py) — each rank prefers its own saved
        # shard; a missing shard (e.g. resuming with a different num_processes) falls back to
        # rank 0's state: correct params, fresh momentum for that rank's Muon shard.
        st_path = os.path.join(args.resume, f"trainer_state_rank{accelerator.process_index}.pt")
        if accelerator.process_index == 0 or not os.path.exists(st_path):
            if accelerator.process_index != 0:
                accelerator.print(f"WARNING: {st_path} missing; rank "
                                  f"{accelerator.process_index} resumes from rank 0's "
                                  f"optimizer state (Muon momentum restarts for its shard)")
            st_path = os.path.join(args.resume, "trainer_state.pt")
        st = torch.load(st_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(st["optimizer"])
        start_step, tokens_seen = st["step"], st["tokens_seen"]
        trainer.stage = Stage(st["stage"])

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "metrics.jsonl")

    if args.wandb:
        try:
            accelerator.init_trackers(args.wandb_project, config=vars(args),
                                      init_kwargs={"wandb": {"name": args.run_name,
                                                             "resume": "allow"}})
        except Exception as e:                           # not logged in / offline / etc.
            accelerator.print(f"WARNING: wandb init failed ({e}); continuing without it "
                              f"(wandb login, or WANDB_MODE=offline, or --no-wandb)")
            args.wandb = False

    accelerator.print(
        f"AUM-Ø train: {n_params / 1e6:.1f}M params on {accelerator.device} | "
        f"{tokens_per_step:,} tok/step x {total_steps:,} steps = {tokens_per_step * total_steps / 1e9:.2f}B tokens\n"
        f"  data: {len(train_ds):,} train / {len(val_ds):,} val windows of {args.seq_len} "
        f"(books split into non-overlapping {args.seq_len}-token chunks)\n"
        f"  silence={'on' if cfg.silence_enabled else 'OFF (evidence-core baseline)'} "
        f"stage-ends={ends} warmup={args.warmup_steps} mp={args.mixed_precision} "
        f"wandb={'on (' + args.wandb_project + ')' if args.wandb else 'off'}"
        + (f"\n  resumed from {args.resume} @ step {start_step} stage {int(trainer.stage)}"
           if args.resume else ""))

    from tqdm.auto import tqdm
    bar = tqdm(total=total_steps, initial=start_step, unit="step", dynamic_ncols=True,
               disable=args.no_tqdm or not accelerator.is_main_process, desc=args.run_name)
    say = bar.write if not bar.disable else accelerator.print

    train_iter, val_iter = cycle(train_loader), cycle(val_loader)
    pred_r2 = float("-inf")
    t0, tokens0 = time.time(), tokens_seen
    model.train()

    for step in range(start_step, total_steps):
        # §12 stage management on the token-budget boundaries; 1->2 additionally needs the R^2
        # gate (checked at eval cadence below) — until it clears, stage 1 simply continues.
        if cfg.silence_enabled:
            if trainer.stage == Stage.EVIDENCE_CORE and step >= ends[0] \
                    and pressure_gate_clear(pred_r2, schedule):
                trainer.maybe_advance_stage(pred_r2)
                say(f"step {step}: stage -> {int(trainer.stage)} (R^2={pred_r2:.3f})")
            elif trainer.stage == Stage.FORCED_REVISION and step >= ends[1]:
                trainer.maybe_advance_stage(pred_r2)
                say(f"step {step}: stage -> {int(trainer.stage)}")
            elif trainer.stage == Stage.SOFT_HALTING and step >= ends[2]:
                trainer.maybe_advance_stage(pred_r2)
                say(f"step {step}: stage -> {int(trainer.stage)}")
            if trainer.stage >= Stage.SOFT_HALTING:      # lambda_C ramp inside stage 3 (§12)
                ramp = (step - ends[1]) / max(1, ends[2] - ends[1])
                trainer.config.lambda_compute = args.lambda_compute_max * min(max(ramp, 0.0), 1.0)

        # LR: shared warmup + cosine, scaled onto each group's base lr (§13)
        scale = lr_scale(step, args.warmup_steps, total_steps)
        for g, base in zip(optimizer.param_groups, base_lrs):
            g["lr"] = base * scale

        p_explore = p_explore_at(step, total_steps, schedule)
        step_metrics = None
        for _ in range(args.grad_accum):                 # one optimizer step per iteration
            with accelerator.accumulate(model):
                batch = next(train_iter)
                step_metrics, _ = trainer.train_step(batch, p_explore=p_explore)
        tokens_seen += tokens_per_step
        if args.empty_cache_every and (step + 1) % args.empty_cache_every == 0:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()                  # stop allocator-cache -> swap creep
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        bar.update(1)
        bar.set_postfix(loss=f"{step_metrics['loss']:.3f}", stage=int(trainer.stage),
                        lr=f"x{scale:.2f}", refresh=False)

        if (step + 1) % args.log_every == 0 and accelerator.is_main_process:
            dt = time.time() - t0
            tps = (tokens_seen - tokens0) / max(dt, 1e-9)
            parts = " ".join(f"{k}={v:.4f}" for k, v in step_metrics["parts"].items())
            say(f"step {step + 1:>6}/{total_steps} stage {int(trainer.stage)} "
                f"loss {step_metrics['loss']:.4f} [{parts}] "
                f"lr x{scale:.3f} {tps / 1e3:.1f}k tok/s")
            with open(log_path, "a") as f:
                json.dump({"step": step + 1, "tokens": tokens_seen, "lr_scale": scale,
                           "p_explore": p_explore, **step_metrics}, f)
                f.write("\n")
            if args.wandb:
                accelerator.log(
                    {"train/loss": step_metrics["loss"],
                     **{f"train/{k}": v for k, v in step_metrics["parts"].items()},
                     "lr_scale": scale, "tokens": tokens_seen, "tokens_per_s": tps,
                     "stage": int(trainer.stage), "p_explore": p_explore,
                     "lambda_compute": trainer.config.lambda_compute,
                     **({"train/benefit_mean": step_metrics["benefit_mean"]}
                        if "benefit_mean" in step_metrics else {})},
                    step=step + 1)
            t0, tokens0 = time.time(), tokens_seen

        if (step + 1) % args.eval_every == 0 or step + 1 == total_steps:
            val_loss = evaluate(model, raw_model, cfg, val_iter, args.eval_batches)
            pred_r2 = trainer.pred_val_r2(next(val_iter)[:, :args.r2_len]) \
                if cfg.silence_enabled else float("-inf")
            if accelerator.num_processes > 1:
                # §12 rank-synchronized gate: val loaders shard per rank, so mean the shard
                # statistics — every rank sees the same numbers and advances stages identically
                vm = torch.tensor([val_loss, pred_r2], device=accelerator.device)
                vm = accelerator.reduce(vm, reduction="mean")
                val_loss, pred_r2 = float(vm[0]), float(vm[1])
            say(f"step {step + 1:>6}: val_loss {val_loss:.4f} "
                f"pred_R^2 {pred_r2:.3f} (gate eta={schedule.eta_r2})")
            if accelerator.is_main_process:
                with open(log_path, "a") as f:
                    json.dump({"step": step + 1, "val_loss": val_loss, "pred_r2": pred_r2}, f)
                    f.write("\n")
                if args.wandb:
                    accelerator.log({"val/loss": val_loss, "val/pred_r2": pred_r2},
                                    step=step + 1)
            model.train()

        if (step + 1) % args.save_every == 0 or step + 1 == total_steps:
            path = save_checkpoint(accelerator, model, optimizer, run_dir, step + 1,
                                   trainer.stage, tokens_seen)
            if accelerator.is_main_process:
                say(f"saved {path}")

    bar.close()
    accelerator.print(f"done: {tokens_seen:,} tokens, final stage {int(trainer.stage)}")
    if args.wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()
