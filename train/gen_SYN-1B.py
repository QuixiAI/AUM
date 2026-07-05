#!/usr/bin/env python
"""Generate and inspect the SYN-1B v1.3 synthetic corpus."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
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

INSPECT_STRATA = (
    "top_age_bin",
    "f3_depth2",
    "f4_f2_distractor",
    "eval_sigma_ev",
)


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


def _load_sidecars(out_dir):
    return [rec for _, rec in _iter_sidecars(out_dir)]


def _instance_ids(out_dir, rec, seq_len=4096):
    path = os.path.join(out_dir, rec["shard"])
    mm = np.memmap(path, dtype=np.uint16, mode="r")
    start = int(rec["window_index"]) * seq_len + int(rec["start_offset"])
    return np.asarray(mm[start:start + int(rec["token_len"])], dtype=np.int64).tolist()


def _walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_strings(value)


def _uses_eval_symbol(rec, alpha):
    eval_symbols = set(alpha.sigma_eval)
    return any(value in eval_symbols for value in _walk_strings(rec))


def _record_max_controlled_gap_bin(rec):
    bins = [
        int(gap["target_age_bin"]) for gap in rec.get("controlled_gaps", [])
        if gap.get("target_age_bin") is not None
    ]
    return max(bins) if bins else None


def _select_stratified_records(records, alpha, seed):
    rng = random.Random(seed)
    predicates = {
        "top_age_bin": lambda r: (_record_max_controlled_gap_bin(r) or -1) >= 8,
        "f3_depth2": lambda r: r.get("family") == "F3" and int(r.get("composition_depth", 0)) >= 2,
        "f4_f2_distractor": lambda r: (
            r.get("family") == "F4"
            and any(d.get("mimics_family") == "F2" for d in r.get("distractors", []))
        ),
        "eval_sigma_ev": lambda r: r.get("split") == "eval" and _uses_eval_symbol(r, alpha),
    }
    selected = []
    missing = []
    used = set()
    for name in INSPECT_STRATA:
        candidates = [r for r in records if predicates[name](r)]
        fresh = [r for r in candidates if r.get("instance_id") not in used]
        pool = fresh or candidates
        if not pool:
            missing.append(name)
            continue
        rec = rng.choice(sorted(pool, key=lambda r: r.get("instance_id", "")))
        used.add(rec.get("instance_id"))
        selected.append((name, rec))
    return selected, missing


def _select_random_records(records, n, seed):
    if n <= 0:
        return []
    rng = random.Random(seed)
    pool = sorted(records, key=lambda r: (r.get("split", ""), r.get("family", ""),
                                          r.get("instance_id", "")))
    n = min(n, len(pool))
    return [("random", rec) for rec in rng.sample(pool, n)]


def _write_inspect_sample(out_dir, alpha, selected, sample_dir, seq_len):
    os.makedirs(sample_dir, exist_ok=True)
    texts = []
    all_ids = []
    sidecars = []
    for reason, rec in selected:
        ids = _instance_ids(out_dir, rec, seq_len)
        text = alpha.tokenizer.decode(ids, clean_up_tokenization_spaces=False).strip()
        texts.append(
            f"[{reason} | {rec['split']} {rec['family']} {rec['instance_id']} "
            f"token_len={rec['token_len']} queries={len(rec.get('queries', []))}]\n{text}"
        )
        all_ids.extend(ids)
        all_ids.append(int(alpha.eos_id))
        item = dict(rec)
        item["inspect_reason"] = reason
        sidecars.append(item)
    sample_text = "\n\n<|endoftext|>\n\n".join(texts)
    text_path = os.path.join(sample_dir, "inspect_sample.txt")
    sidecar_path = os.path.join(sample_dir, "inspect_sample.sidecar.json")
    token_path = os.path.join(sample_dir, "inspect_sample.token_ids.json")
    meta_path = os.path.join(sample_dir, "inspect_sample.meta.json")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(sample_text + ("\n" if sample_text else ""))
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecars, f, indent=2, sort_keys=True)
    with open(token_path, "w", encoding="utf-8") as f:
        json.dump(all_ids, f)
    meta = {
        "source_out_dir": out_dir,
        "instances": len(sidecars),
        "inspect_reasons": [reason for reason, _ in selected],
        "families": {
            family: sum(1 for _, rec in selected if rec.get("family") == family)
            for family in sorted({rec.get("family") for _, rec in selected})
        },
        "instance_tokens": sum(int(rec["token_len"]) for _, rec in selected),
        "separator_tokens": len(selected),
        "total_tokens_with_separators": len(all_ids),
        "elided": [],
        "files": {"text": text_path, "sidecar": sidecar_path, "token_ids": token_path},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(json.dumps({"sample": meta, "meta_path": meta_path}, indent=2, sort_keys=True))


def _print_selected_instances(out_dir, alpha, selected, seq_len, text_chars):
    for reason, rec in selected:
        ids = _instance_ids(out_dir, rec, seq_len)
        label_by_pos = {}
        for lab in rec.get("label_positions", []):
            label_by_pos.setdefault(lab["pos"], []).append(f"{lab['role']}={lab['expected']}")
        decoded_text = alpha.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        if text_chars and text_chars > 0:
            decoded_text = decoded_text[:text_chars]
        print("\n" + "=" * 100)
        print(f"{reason} | {rec['split']} {rec['family']} {rec['instance_id']} "
              f"len={rec['token_len']} {rec['shard']}:{rec['window_index']}+{rec['start_offset']}")
        print("- decoded text -")
        print(decoded_text)
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
        print("- sidecar excerpt policy -")
        print("stdout shows decoded text and label alignment only; full sidecars are written "
              "when --inspect-out-dir is set.")


def inspect_random_or_stratified(out_dir, alpha, random_n, stratified, seed, seq_len=4096,
                                 text_chars=1600, sample_dir=None):
    records = _load_sidecars(out_dir)
    selected = []
    missing = []
    if stratified:
        strata_selected, missing = _select_stratified_records(records, alpha, seed)
        selected.extend(strata_selected)
    selected.extend(_select_random_records(records, random_n, seed))
    if not selected:
        raise SystemExit("no records selected for inspection")
    _print_selected_instances(out_dir, alpha, selected, seq_len, text_chars)
    if missing:
        print(json.dumps({"missing_inspect_strata": missing}, indent=2), file=sys.stderr)
    if sample_dir:
        _write_inspect_sample(out_dir, alpha, selected, sample_dir, seq_len)


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
                            f"{rec['family']}:predicts={q['answer']}")

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
    ap = argparse.ArgumentParser(description="Generate SYN-1B v1.3 synthetic shards.")
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
    ap.add_argument("--inspect-random", type=int, default=0,
                    help="print N uniformly random aligned instances from --out-dir")
    ap.add_argument("--inspect-strata", action="store_true",
                    help="print seeded audit-stratum instances: top age bin, F3 depth-2, "
                         "F4/F2 distractor, and eval Sigma_ev")
    ap.add_argument("--inspect-seed", type=int, default=20240705,
                    help="seed for --inspect-random and --inspect-strata")
    ap.add_argument("--inspect-text-chars", type=int, default=1600,
                    help="decoded text chars to print per inspected instance; <=0 prints full text")
    ap.add_argument("--inspect-out-dir", default=None,
                    help="write full decoded inspected instances and full sidecars to this dir")
    ap.add_argument("--inspect-window", type=int, default=0,
                    help="print N decoded packed windows with aligned labels from --out-dir")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    tok, alpha = build_real_alphabet(args.tokenizer, args.seed)
    if args.inspect:
        return inspect(args.out_dir, alpha, args.inspect, args.seq_len)
    if args.inspect_random or args.inspect_strata:
        return inspect_random_or_stratified(
            args.out_dir, alpha, args.inspect_random, args.inspect_strata,
            args.inspect_seed, args.seq_len, args.inspect_text_chars, args.inspect_out_dir,
        )
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
