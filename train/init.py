#!/usr/bin/env python
"""Initialize AUM-Ø-Tiny v5.3 (~78M) — random weights per the AUM-Ø.md recipe (Appendix A).

Unlike upstream Mamba, AUM keeps its training code in ./train. This is step one: materialize a
fresh, randomly-initialized checkpoint for the reference config so later training scripts start
from a well-defined point.

What it builds
--------------
The reference ``AumConfig`` IS the Appendix-A physical layout: d_model=512, 12 evidence layers,
GQA 8/2 x 64, U-phase 4 x 128, SwiGLU 1408, sigma=128, vocab=49152, tied embeddings. Instantiating
``AumLMHeadModel`` applies the init recipe:

  - token embedding      ~ N(0, 0.02)   (tied to the output classifier)
  - Linear weights       ~ PyTorch kaiming_uniform(a=sqrt(5)); residual output projections
                           (unfold.out_proj, ground_attn.o_proj, mlp.down_proj) are re-kaiming'd
                           then scaled by 1/sqrt(2L) (GPT-2 / Megatron prenorm-residual scheme)
  - biases               -> 0            (most AUM projections are bias-free)
  - per-module priors set in each module's __init__ and preserved: unfold.dt_bias =
    inverse-softplus(dt), dt ~ logU[1e-3, 0.1]; unfold.A_log = 0; unfold.D = 1; every RMSNorm and
    the gated-readout weight = 1.

Total ~78.3M (silence block ~1.77M; the silence-ablated evidence core ~76.5M) — matching AUM-Ø.md.

Output
------
Writes ``<out>/config.json`` + ``<out>/pytorch_model.bin`` (the AumLMHeadModel.save_pretrained
format, so ``AumLMHeadModel.from_pretrained(<out>)`` loads it) plus ``<out>/init_manifest.json``
(seed, dtype, param counts, git commit, spec target). Run:

    python train/init.py                      # fp32 master weights, seed 0
    python train/init.py --dtype bfloat16      # Appendix-A dtypes (A_log / unfold.norm stay F32)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import torch

# Make `aum_ssm` importable when run as `python train/init.py` from the repo root or elsewhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aum_ssm.models.config_aum import AumConfig
from aum_ssm.models.aum_lm import AumLMHeadModel

SPEC_TARGET_M = 78.0                              # AUM-Ø.md Appendix A: total ~= 78M
# Appendix-A tensors that stay F32 even in a bf16 export (the SSM decay base + gated-readout norm).
_F32_KEEP = ("A_log", "unfold.norm.weight")


def build_model(seed: int = 0, **cfg_overrides):
    """The reference AUM-Ø-Tiny v5.3 model (full silence block present), randomly initialized."""
    torch.manual_seed(seed)
    cfg = AumConfig(silence_enabled=True, **cfg_overrides)
    model = AumLMHeadModel(cfg)
    return model, cfg


def param_report(model):
    """(total unique params, {bucket: numel}) — tied embed/classifier counted once."""
    seen, total, buckets = set(), 0, defaultdict(int)
    for name, p in model.named_parameters():         # remove_duplicate=True -> tied weight once
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if "silence" in name:
            key = "backbone.silence"
        elif ".layers." in name:
            key = "backbone.layers[0-11]"
        elif "embedding" in name:
            key = "backbone.embedding (tied classifier)"
        else:
            key = "backbone.norm_f"
        buckets[key] += p.numel()
    return total, dict(buckets)


def cast_state_dict(state_dict, dtype):
    """Cast floating tensors to `dtype`, keeping the Appendix-A F32 tensors (A_log, unfold.norm)."""
    if dtype == torch.float32:
        return state_dict
    out = {}
    for k, v in state_dict.items():
        keep_f32 = v.dtype == torch.float32 and any(tag in k for tag in _F32_KEEP)
        out[k] = v if (keep_f32 or not v.is_floating_point()) else v.to(dtype)
    return out


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


@torch.no_grad()
def _forward_sanity(model, vocab_size, seed):
    """A tiny forward pass on random tokens must give finite logits of the right shape."""
    torch.manual_seed(seed + 1)
    ids = torch.randint(0, vocab_size, (1, 8))
    logits = model.eval()(ids).logits
    assert logits.shape == (1, 8, vocab_size), logits.shape
    assert torch.isfinite(logits).all(), "non-finite logits from a freshly initialized model"
    return float(logits.float().std())


def main():
    ap = argparse.ArgumentParser(description="Initialize AUM-Ø-Tiny v5.3 (~78M) random weights.")
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "train", "checkpoints",
                                                  "aum-tiny-v5.3-init"),
                    help="output checkpoint directory")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible weights")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                    help="storage dtype (bfloat16 follows Appendix A: A_log / unfold.norm stay F32)")
    ap.add_argument("--vocab-size", type=int, default=None, help="override vocab (default 49152)")
    ap.add_argument("--no-check", action="store_true", help="skip the forward-pass sanity check")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty --out")
    args = ap.parse_args()

    if os.path.isdir(args.out) and os.listdir(args.out) and not args.force:
        ap.error(f"{args.out} exists and is non-empty; pass --force to overwrite")

    overrides = {} if args.vocab_size is None else {"vocab_size": args.vocab_size}
    model, cfg = build_model(seed=args.seed, **overrides)
    total, buckets = param_report(model)
    emb = buckets.get("backbone.embedding (tied classifier)", 0)

    print(f"AUM-Ø-Tiny v5.3   seed={args.seed}  storage={args.dtype}")
    print(f"  total params : {total / 1e6:8.3f} M   (spec target ~= {SPEC_TARGET_M} M)")
    for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"    {k:34s} {v / 1e6:8.3f} M")
    print(f"  non-embedding: {(total - emb) / 1e6:8.3f} M")
    print(f"  silence block: {buckets.get('backbone.silence', 0) / 1e6:8.3f} M   "
          f"(evidence-core baseline ~= {(total - buckets.get('backbone.silence', 0)) / 1e6:.3f} M)")
    assert 76.0 <= total / 1e6 <= 80.0, f"param count {total / 1e6:.2f}M is off spec (~78M)"

    if not args.no_check:
        std = _forward_sanity(model, cfg.vocab_size, args.seed)
        print(f"  forward check: OK (logit std {std:.3f}, all finite)")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    os.makedirs(args.out, exist_ok=True)
    sd = cast_state_dict(model.state_dict(), dtype)
    if cfg.tie_embeddings and "lm_head.weight" in sd and "backbone.embedding.weight" in sd:
        sd["lm_head.weight"] = sd["backbone.embedding.weight"]   # keep the tie (one stored copy)
    weights_path = os.path.join(args.out, "pytorch_model.bin")
    torch.save(sd, weights_path)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
    manifest = {
        "model": "AUM-Ø-Tiny v5.3",
        "seed": args.seed,
        "storage_dtype": args.dtype,
        "params_total": total,
        "params_total_M": round(total / 1e6, 4),
        "params_by_bucket": buckets,
        "spec_target_M": SPEC_TARGET_M,
        "git_commit": _git_commit(),
        "created_unix": int(time.time()),
        "config": cfg.__dict__,
    }
    with open(os.path.join(args.out, "init_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    mb = os.path.getsize(weights_path) / 1e6
    print(f"  wrote {args.out}")
    print(f"        pytorch_model.bin ({mb:.1f} MB), config.json, init_manifest.json")
    print(f"  load with: AumLMHeadModel.from_pretrained({args.out!r})")


if __name__ == "__main__":
    main()
