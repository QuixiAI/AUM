#!/usr/bin/env python
"""Run a model-side plumbing probe on a packed SYN window.

This is not an evaluation of SYN task learning. It verifies that the real model input path,
tokenizer, causal shift, and sidecar positions agree on the tokens used by the harness.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aum_ssm.models.aum_lm import AumLMHeadModel
from train.syn.alphabet import MODEL_TOKENIZER
from train.tokenizer import load_tokenizer


def _iter_sidecars(out_dir, split):
    path = os.path.join(out_dir, "sidecars", f"{split}.jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _select_window(records, attn_window, split, shard, window_index):
    if shard is not None and window_index is not None:
        return shard, int(window_index)
    hard = next((
        r for r in records
        if any(int(q.get("minimal_sufficient_suffix_len", 0)) > attn_window
               for q in r.get("queries", []))
    ), None)
    if hard is not None:
        return hard["shard"], int(hard["window_index"])
    first = records[0]
    return first["shard"], int(first["window_index"])


def _nearest_non_query_position(pos, query_positions, ids, eos_id):
    for radius in range(1, 32):
        for candidate in (pos - radius, pos + radius):
            if 0 <= candidate < len(ids) - 1:
                if candidate not in query_positions and int(ids[candidate + 1]) != eos_id:
                    return candidate
    return None


def run_probe(args):
    tokenizer = load_tokenizer(MODEL_TOKENIZER)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AumLMHeadModel.from_pretrained(args.checkpoint, device=device, dtype=dtype).eval()
    attn_window = int(getattr(model.config, "attn_window", 0) or 0)

    records = list(_iter_sidecars(args.out_dir, args.split))
    if not records:
        raise SystemExit(f"no sidecar records for split={args.split}")
    shard, window_index = _select_window(records, attn_window, args.split, args.shard,
                                         args.window_index)
    window_records = [
        r for r in records if r["shard"] == shard and int(r["window_index"]) == window_index
    ]

    path = os.path.join(args.out_dir, shard)
    mm = np.memmap(path, dtype=np.uint16, mode="r")
    start = window_index * args.seq_len
    ids = np.asarray(mm[start:start + args.seq_len], dtype=np.int64)
    if ids.size != args.seq_len:
        raise SystemExit(f"short window: got {ids.size}, expected {args.seq_len}")

    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        output = model(input_ids)
    logits = output.logits
    next_targets = input_ids[:, 1:]
    next_logits = logits[:, :-1, :]
    losses = F.cross_entropy(next_logits.reshape(-1, next_logits.size(-1)),
                             next_targets.reshape(-1), reduction="none").reshape(1, -1)

    eos_id = int(tokenizer.eos_token_id)
    non_eos_mask = (input_ids[:, 1:] != eos_id)
    non_eos_losses = losses[non_eos_mask]
    query_positions = set()
    rows = []
    bad = []
    for record in window_records:
        base = int(record["start_offset"])
        expected_anchor = ":" if record["family"] == "F5" else "->"
        for query_index, query in enumerate(record.get("queries", [])):
            pred_pos = base + int(query["answer_pos"]) - 1
            answer_pos = base + int(query["answer_pos"])
            suffix_len = query.get("minimal_sufficient_suffix_len")
            beyond_window = (
                suffix_len is not None and attn_window > 0 and int(suffix_len) > attn_window
            )
            query_positions.add(pred_pos)
            if pred_pos < 0 or answer_pos >= args.seq_len:
                bad.append({"instance_id": record["instance_id"], "query_index": query_index,
                            "error": "query outside selected window"})
                continue
            pred_decoded = tokenizer.decode([int(ids[pred_pos])],
                                            clean_up_tokenization_spaces=False).strip()
            answer_decoded = tokenizer.decode([int(ids[answer_pos])],
                                              clean_up_tokenization_spaces=False).strip()
            anchor_ok = pred_decoded == expected_anchor
            answer_ok = answer_decoded == query["answer"]
            loss = float(losses[0, pred_pos].detach().cpu())
            finite_loss = bool(np.isfinite(loss))
            if not anchor_ok or not answer_ok or not finite_loss:
                bad.append({"instance_id": record["instance_id"], "query_index": query_index,
                            "pred_pos": pred_pos, "answer_pos": answer_pos,
                            "pred_decoded": pred_decoded, "expected_anchor": expected_anchor,
                            "answer_decoded": answer_decoded, "expected_answer": query["answer"],
                            "loss": loss})
            rows.append({
                "instance_id": record["instance_id"],
                "family": record["family"],
                "query_index": query_index,
                "pred_pos": pred_pos,
                "answer_pos": answer_pos,
                "anchor": pred_decoded,
                "anchor_ok": anchor_ok,
                "answer": query["answer"],
                "answer_decoded": answer_decoded,
                "answer_ok": answer_ok,
                "loss": loss,
                "finite_loss": finite_loss,
                "hard_suffix": bool(query.get("hard_suffix", False)),
                "beyond_window": bool(beyond_window),
                "minimal_sufficient_suffix_len": suffix_len,
            })

    matched_rows = []
    for row in rows:
        match_pos = _nearest_non_query_position(row["pred_pos"], query_positions, ids, eos_id)
        if match_pos is None:
            continue
        matched_rows.append({
            "matched_pos": match_pos,
            "near_query_pred_pos": row["pred_pos"],
            "loss": float(losses[0, match_pos].detach().cpu()),
            "token": tokenizer.decode([int(ids[match_pos])], clean_up_tokenization_spaces=False),
            "next_token": tokenizer.decode([int(ids[match_pos + 1])],
                                           clean_up_tokenization_spaces=False),
        })

    def mean(values):
        return float(np.mean(values)) if values else None

    by_group = defaultdict(list)
    by_window = defaultdict(list)
    for row in rows:
        by_group["hard" if row["hard_suffix"] else "non_hard"].append(row["loss"])
        by_window["beyond_window" if row["beyond_window"] else "within_window"].append(row["loss"])
        by_group[row["family"]].append(row["loss"])

    report = {
        "ok": not bad and bool(rows),
        "checkpoint": args.checkpoint,
        "checkpoint_note": args.checkpoint_note,
        "out_dir": args.out_dir,
        "split": args.split,
        "shard": shard,
        "window_index": window_index,
        "seq_len": args.seq_len,
        "device": device,
        "dtype": str(dtype),
        "attn_window": attn_window,
        "instance_count": len(window_records),
        "query_positions_checked": len(rows),
        "anchor_answer_failures": bad,
        "loss_mean_all_targets": float(losses.mean().detach().cpu()),
        "loss_mean_non_eos_targets": float(non_eos_losses.mean().detach().cpu()) if non_eos_losses.numel() else None,
        "non_eos_target_count": int(non_eos_mask.sum().detach().cpu()),
        "eos_target_count": int((~non_eos_mask).sum().detach().cpu()),
        "query_loss_mean": mean([r["loss"] for r in rows]),
        "matched_non_query_loss_mean": mean([r["loss"] for r in matched_rows]),
        "query_loss_by_group": {k: mean(v) for k, v in sorted(by_group.items())},
        "query_loss_by_beyond_window": {k: mean(v) for k, v in sorted(by_window.items())},
        "hard_query_count": sum(r["hard_suffix"] for r in rows),
        "hard_suffix_query_count": sum(r["hard_suffix"] for r in rows),
        "beyond_window_query_count": sum(r["beyond_window"] for r in rows),
        "rows": rows,
        "matched_non_query_rows": matched_rows,
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def main():
    parser = argparse.ArgumentParser(description="Probe model/sidecar alignment on SYN data.")
    parser.add_argument("--out-dir", default="/tmp/aum_syn_dry_v13")
    parser.add_argument("--checkpoint", default="train/checkpoints/aum-tiny-v6-cuda-1b/latest")
    parser.add_argument("--checkpoint-note",
                        default="This checkpoint predates SYN-1B v1.3; losses are plumbing diagnostics, not task-learning evidence.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--shard", default=None)
    parser.add_argument("--window-index", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report", default="/tmp/aum_syn_forward_probe/report.json")
    args = parser.parse_args()
    raise SystemExit(run_probe(args))


if __name__ == "__main__":
    main()
