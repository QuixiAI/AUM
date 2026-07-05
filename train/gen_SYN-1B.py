#!/usr/bin/env python
"""Generate and inspect the SYN-1B v1.1 synthetic corpus."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train.syn.alphabet import MODEL_TOKENIZER, build_alphabet
from train.syn.pack import run_generation
from train.syn.qa import run_qa
from train.syn.registry import Registry


def build_real_alphabet(tokenizer_name, seed):
    from train.tokenizer import load_tokenizer, verify
    tok = load_tokenizer(tokenizer_name)
    verify(tok, 49152)
    alpha = build_alphabet(tok, seed)
    ident = alpha.identity
    print(
        "SYN tokenizer: "
        f"name={ident.name_or_path} class={ident.tokenizer_class} "
        f"transformers={ident.transformers_version} vocab={ident.vocab_size} "
        f"backend_hash={ident.backend_hash[:16]}",
        file=sys.stderr,
    )
    return tok, alpha


def self_test():
    tok, alpha = build_real_alphabet(MODEL_TOKENIZER, 1)
    tmp = tempfile.mkdtemp(prefix="aum_syn_selftest_")
    try:
        reg = Registry(seed=1)
        summary = run_generation(alpha, reg, tmp, MODEL_TOKENIZER, len(tok), train_tokens=80_000,
                                 eval_tokens=32_768, shard_size_tokens=65_536, seed=1)
        report = run_qa(tmp, alpha)
        if not report["ok"]:
            raise SystemExit(json.dumps(report, indent=2))
        print(json.dumps({"out_dir": tmp, "summary": summary, "qa": report}, indent=2))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _iter_sidecars(out_dir):
    for split in ("train", "eval"):
        path = os.path.join(out_dir, "sidecars", f"{split}.jsonl.gz")
        if not os.path.exists(path):
            continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield split, json.loads(line)


def inspect(out_dir, alpha, n, seq_len=4096):
    shown = defaultdict(int)
    for split, rec in _iter_sidecars(out_dir):
        fam = rec["family"]
        if shown[fam] >= n:
            continue
        path = os.path.join(out_dir, rec["shard"])
        mm = np.memmap(path, dtype=np.uint16, mode="r")
        start = rec["window_index"] * seq_len + rec["start_offset"]
        ids = np.asarray(mm[start:start + rec["token_len"]], dtype=np.int64).tolist()
        label_by_pos = {}
        for lab in rec.get("label_positions", []):
            label_by_pos.setdefault(lab["pos"], []).append(f"{lab['role']}={lab['expected']}")
        print("\n" + "=" * 100)
        print(f"{split} {fam} {rec['instance_id']} len={rec['token_len']} "
              f"{rec['shard']}:{rec['window_index']}+{rec['start_offset']}")
        print("- decoded text -")
        print(alpha.tokenizer.decode(ids, clean_up_tokenization_spaces=False)[:1600])
        print("- token_index | decoded_token | label_at_this_position -")
        interesting = sorted(label_by_pos)
        rows = []
        for p in interesting:
            rows.extend(range(max(0, p - 2), min(len(ids), p + 3)))
        rows = sorted(set(rows)) if rows else list(range(min(len(ids), 80)))
        prev = None
        for i in rows:
            if prev is not None and i > prev + 1:
                print("  ... | ...                | ...")
            decoded = alpha.tokenizer.decode([ids[i]], clean_up_tokenization_spaces=False)
            label = "; ".join(label_by_pos.get(i, []))
            print(f"{i:5d} | {decoded!r:18s} | {label}")
            prev = i
        shown[fam] += 1
        if all(shown[f] >= n for f in ("F1", "F2", "F3", "F4", "F5")):
            break


def inspect_windows(out_dir, alpha, n, seq_len=4096):
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    by_window = defaultdict(list)
    for split, rec in _iter_sidecars(out_dir):
        by_window[(split, rec["shard"], rec["window_index"])].append(rec)

    shown = 0
    for split in ("train", "eval"):
        if split not in manifest.get("splits", {}):
            continue
        for shard in manifest["splits"][split]["shards"]:
            path = os.path.join(out_dir, shard["name"])
            mm = np.memmap(path, dtype=np.uint16, mode="r")
            for window_index in range(int(shard["n_tokens"]) // seq_len):
                if shown >= n:
                    return
                start = window_index * seq_len
                ids = np.asarray(mm[start:start + seq_len], dtype=np.int64).tolist()
                labels = defaultdict(list)
                for rec in by_window.get((split, shard["name"], window_index), []):
                    base = int(rec["start_offset"])
                    for lab in rec.get("label_positions", []):
                        labels[base + lab["pos"]].append(
                            f"{rec['family']}:{lab['role']}={lab['expected']}")
                    for q in rec.get("queries", []):
                        labels[base + q["answer_pos"] - 1].append(
                            f"{rec['family']}:predicts={q.get('answer', q.get('correct_answer'))}")

                print("\n" + "=" * 100)
                print(f"{split} window {shard['name']}:{window_index} labels={sum(len(v) for v in labels.values())}")
                print("- decoded window prefix -")
                print(alpha.tokenizer.decode(ids, clean_up_tokenization_spaces=False)[:3000])
                print("- token_index | decoded_token | label_at_this_position -")
                rows = []
                for p in sorted(labels):
                    rows.extend(range(max(0, p - 2), min(len(ids), p + 3)))
                prev = None
                for i in sorted(set(rows)):
                    if prev is not None and i > prev + 1:
                        print("  ... | ...                | ...")
                    decoded = alpha.tokenizer.decode([ids[i]], clean_up_tokenization_spaces=False)
                    print(f"{i:5d} | {decoded!r:18s} | {'; '.join(labels.get(i, []))}")
                    prev = i
                shown += 1


def main():
    ap = argparse.ArgumentParser(description="Generate SYN-1B v1.1 synthetic shards.")
    ap.add_argument("--out-dir", default=os.path.join("train", "data_syn"))
    ap.add_argument("--tokenizer", default=MODEL_TOKENIZER)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--shard-size-tokens", type=int, default=100_000_000)
    ap.add_argument("--train-tokens", type=int, default=1_000_000_000)
    ap.add_argument("--eval-tokens", type=int, default=40_000_000)
    ap.add_argument("--dry-run-tokens", type=int, default=0,
                    help="override train/eval token budgets for a small validation run")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--qa-only", default=None, help="run QA against an existing generated dir")
    ap.add_argument("--inspect", type=int, default=0,
                    help="print N aligned instances per family from --out-dir")
    ap.add_argument("--inspect-window", type=int, default=0,
                    help="print N decoded packed windows with aligned labels from --out-dir")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    tok, alpha = build_real_alphabet(args.tokenizer, args.seed)
    if args.inspect:
        return inspect(args.out_dir, alpha, args.inspect, args.seq_len)
    if args.inspect_window:
        return inspect_windows(args.out_dir, alpha, args.inspect_window, args.seq_len)
    if args.qa_only:
        report = run_qa(args.qa_only, alpha, args.seq_len)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["ok"] else 1)

    train_tokens = args.dry_run_tokens or args.train_tokens
    eval_tokens = max(args.seq_len * 5, (args.dry_run_tokens // 4) if args.dry_run_tokens else args.eval_tokens)
    reg = Registry(seed=args.seed)
    summary = run_generation(
        alpha, reg, args.out_dir, args.tokenizer, len(tok), seq_len=args.seq_len,
        shard_size_tokens=args.shard_size_tokens, train_tokens=train_tokens,
        eval_tokens=eval_tokens, seed=args.seed,
    )
    report = run_qa(args.out_dir, alpha, args.seq_len)
    print(json.dumps({"summary": summary, "qa": report}, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
