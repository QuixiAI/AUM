"""SYN-1B generator QA checks."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict

import numpy as np

from train.syn.alphabet import STRUCTURAL_TOKENS
from train.syn.harness_readers import open_jsonl, prediction_pos_for_answer
from train.syn.pack import (MAX_INSTANCES_PER_WINDOW, WINDOW_TASK_FRACTION_QA,
                            WINDOW_TASK_FRACTION_TARGET)


def _records(out_dir, split):
    path = os.path.join(out_dir, "sidecars", f"{split}.jsonl.gz")
    return list(open_jsonl(path))


def _manifest(out_dir):
    path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _iter_shard_windows(out_dir, split, seq_len=4096):
    manifest = _manifest(out_dir)
    if manifest is None or split not in manifest.get("splits", {}):
        return
    for shard in manifest["splits"][split].get("shards", []):
        name = shard["name"]
        mm = np.memmap(os.path.join(out_dir, name), dtype=np.uint16, mode="r")
        for window_index in range(mm.size // seq_len):
            start = window_index * seq_len
            yield name, window_index, np.asarray(mm[start:start + seq_len], dtype=np.uint16)


def _records_by_window(records):
    by_window = defaultdict(list)
    for r in records:
        by_window[(r["shard"], r["window_index"])].append(r)
    return by_window


def tokenizer_audit(alpha):
    missing = [w for w in alpha.sigma + tuple(alpha.bg_vocab) if w not in alpha.token_to_id]
    return {"name": "tokenizer", "ok": not missing, "missing": missing[:20],
            "sigma": len(alpha.sigma), "bg_vocab": len(alpha.bg_vocab)}


def manifest_tokenizer_audit(out_dir, alpha):
    if alpha is None or alpha.identity is None:
        return {"name": "manifest_tokenizer", "ok": False, "error": "alpha identity required"}
    manifest = _manifest(out_dir)
    if manifest is None:
        return {"name": "manifest_tokenizer", "ok": False, "error": "manifest.json missing"}
    ident = manifest.get("tokenizer_identity") or {}
    expected = alpha.identity.__dict__
    keys = ("name_or_path", "vocab_size", "length", "backend_hash", "vocab_hash")
    mismatches = {k: {"manifest": ident.get(k), "loaded": expected.get(k)}
                  for k in keys if ident.get(k) != expected.get(k)}
    if manifest.get("vocab_size") != len(alpha.tokenizer):
        mismatches["manifest.vocab_size"] = {
            "manifest": manifest.get("vocab_size"),
            "loaded": len(alpha.tokenizer),
        }
    return {"name": "manifest_tokenizer", "ok": not mismatches,
            "mismatches": mismatches, "tokenizer": manifest.get("tokenizer")}


def packing_policy_audit(out_dir, seq_len=4096):
    manifest = _manifest(out_dir)
    if manifest is None:
        return {"name": "packing_policy", "ok": False, "error": "manifest.json missing"}
    policy = manifest.get("packing_policy") or {}
    expected = {
        "unit": "fixed_window",
        "window_tokens": seq_len,
        "max_synthetic_instances_per_window": MAX_INSTANCES_PER_WINDOW,
        "window_background": "eos_padding",
        "window_task_fraction_target": list(WINDOW_TASK_FRACTION_TARGET),
        "window_task_fraction_qa": list(WINDOW_TASK_FRACTION_QA),
        "instance_boundary_mask": False,
        "eos_between_instances": True,
    }
    mismatches = {k: {"manifest": policy.get(k), "expected": v}
                  for k, v in expected.items() if policy.get(k) != v}
    return {"name": "packing_policy", "ok": not mismatches,
            "mismatches": mismatches, "policy": policy}


def distribution_audit(records):
    lengths = [r["token_len"] for r in records]
    bad_len = [x for x in lengths if x < 48 or x > 3968]
    age_bins = Counter(q.get("age_bin") for r in records for q in r.get("queries", [])
                       if q.get("age_bin") is not None)
    event_fracs = []
    for r in records:
        if r.get("family") == "F3":
            continue
        for e in r.get("events", []):
            event_fracs.append(e["pos"] / max(1, r["token_len"]))
    bad_events = [x for x in event_fracs if x < 0.10 or x > 0.90]
    return {"name": "distribution", "ok": not bad_len and not bad_events,
            "records": len(records), "bad_len": len(bad_len), "bad_events": len(bad_events),
            "age_bins": dict(age_bins)}


def marker_audit(records):
    eventful = sum(bool(r.get("events")) for r in records)
    nulls = sum(r.get("family") == "F4" for r in records)
    return {"name": "marker", "ok": eventful > 0 and nulls > 0,
            "eventful_records": eventful, "null_records": nulls}


def replay_audit(records):
    seen = set()
    dup = 0
    for r in records:
        h = r.get("token_hash")
        if h in seen:
            dup += 1
        seen.add(h)
    return {"name": "dedup", "ok": dup == 0, "duplicates": dup, "hashes": len(seen)}


def window_integrity(out_dir, split, records, seq_len=4096):
    by_shard = {}
    for r in records:
        by_shard.setdefault(r["shard"], []).append(r)
    bad = 0
    crowded = 0
    for shard, rs in by_shard.items():
        path = os.path.join(out_dir, shard)
        mm = np.memmap(path, dtype=np.uint16, mode="r")
        per_window = Counter(r["window_index"] for r in rs)
        crowded += sum(1 for c in per_window.values() if c > MAX_INSTANCES_PER_WINDOW)
        for r in rs:
            start = r["window_index"] * seq_len + r["start_offset"]
            end = start + r["token_len"]
            if r["start_offset"] < 0 or r["start_offset"] + r["token_len"] > seq_len or end > mm.size:
                bad += 1
                continue
            h = hashlib.sha256(np.asarray(mm[start:end], dtype=np.uint16).tobytes()).hexdigest()
            if h != r.get("token_hash"):
                bad += 1
    return {"name": f"window_integrity_{split}", "ok": bad == 0 and crowded == 0,
            "bad_records": bad, "crowded_windows": crowded}


def task_density_audit(out_dir, split, records, seq_len=4096):
    lo, hi = WINDOW_TASK_FRACTION_QA
    by_window = _records_by_window(records)
    window_fracs = []
    instance_fracs = []
    bad_windows = []
    bad_records = []
    total_task = 0
    total_tokens = 0
    for r in records:
        task = int(r.get("task_token_count", -1))
        filler = int(r.get("filler_token_count", -1))
        token_len = int(r.get("token_len", 0))
        if task < 0 or filler < 0 or task + filler != token_len:
            bad_records.append({"instance_id": r.get("instance_id"), "task": task,
                                "filler": filler, "token_len": token_len})
            continue
        instance_fracs.append(task / max(1, token_len))
    for shard, window_index, _ in _iter_shard_windows(out_dir, split, seq_len) or []:
        task = sum(int(r.get("task_token_count", 0))
                   for r in by_window.get((shard, window_index), []))
        frac = task / seq_len
        window_fracs.append(frac)
        total_task += task
        total_tokens += seq_len
        if frac < lo or frac > hi:
            bad_windows.append({"shard": shard, "window_index": window_index,
                                "task_fraction": frac, "task_tokens": task})
    corpus_frac = total_task / max(1, total_tokens)
    return {"name": f"task_density_{split}",
            "ok": bool(window_fracs) and not bad_records and not bad_windows and lo <= corpus_frac <= hi,
            "window_fraction_min": min(window_fracs) if window_fracs else None,
            "window_fraction_mean": float(np.mean(window_fracs)) if window_fracs else None,
            "window_fraction_max": max(window_fracs) if window_fracs else None,
            "instance_fraction_mean": float(np.mean(instance_fracs)) if instance_fracs else None,
            "corpus_task_fraction": corpus_frac,
            "target": {"min": lo, "max": hi},
            "bad_windows": bad_windows[:20],
            "bad_records": bad_records[:20]}


def packed_window_roundtrip(out_dir, split, alpha, seq_len=4096):
    if alpha is None:
        return {"name": f"packed_window_roundtrip_{split}", "ok": False,
                "error": "alpha required"}
    bad = []
    checked = 0
    for shard, window_index, ids_arr in _iter_shard_windows(out_dir, split, seq_len) or []:
        checked += 1
        ids = ids_arr.astype(np.int64).tolist()
        text = alpha.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        ids2 = alpha.tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids2 != ids:
            first = next((i for i, pair in enumerate(zip(ids, ids2)) if pair[0] != pair[1]),
                         min(len(ids), len(ids2)))
            bad.append({"shard": shard, "window_index": window_index,
                        "first_mismatch": first, "len_ids": len(ids),
                        "len_retoked": len(ids2)})
            if len(bad) >= 20:
                break
    return {"name": f"packed_window_roundtrip_{split}", "ok": checked > 0 and not bad,
            "windows": checked, "bad": bad}


def window_background_scaffolding_audit(out_dir, split, records, alpha, seq_len=4096):
    if alpha is None:
        return {"name": f"window_background_scaffolding_{split}", "ok": False,
                "error": "alpha required"}
    structural_ids = {alpha.single_id(w): w for w in STRUCTURAL_TOKENS if w in alpha.token_to_id}
    by_window = _records_by_window(records)
    bad = Counter()
    checked_background = 0
    for shard, window_index, ids in _iter_shard_windows(out_dir, split, seq_len) or []:
        covered = np.zeros(seq_len, dtype=bool)
        for r in by_window.get((shard, window_index), []):
            start = int(r["start_offset"])
            end = min(seq_len, start + int(r["token_len"]))
            if 0 <= start < end:
                covered[start:end] = True
        bg_positions = np.flatnonzero(~covered)
        checked_background += int(bg_positions.size)
        for pos in bg_positions:
            word = structural_ids.get(int(ids[pos]))
            if word is not None:
                bad[word] += 1
    return {"name": f"window_background_scaffolding_{split}", "ok": not bad,
            "background_tokens": checked_background, "bad_counts": dict(bad)}


def label_audit(records):
    bad = 0
    total = 0
    for r in records:
        for q in r.get("queries", []):
            total += 1
            p = prediction_pos_for_answer(q)
            if p < 0 or p >= r["token_len"] - 1:
                bad += 1
    return {"name": "labels", "ok": bad == 0 and total > 0, "queries": total, "bad": bad}


def query_position_contract(out_dir, records, alpha, seq_len=4096):
    if alpha is None:
        return {"name": "query_position_contract", "ok": False, "error": "alpha required"}
    bad = []
    checked = 0
    maps = {}
    for r in records:
        path = os.path.join(out_dir, r["shard"])
        mm = maps.get(path)
        if mm is None:
            mm = maps[path] = np.memmap(path, dtype=np.uint16, mode="r")
        base = r["window_index"] * seq_len + r["start_offset"]
        expected_pred = ":" if r.get("family") == "F5" else "->"
        for q in r.get("queries", []):
            checked += 1
            answer_pos = int(q["answer_pos"])
            pred_pos = prediction_pos_for_answer(q)
            answer_word = q.get("answer", q.get("correct_answer"))
            if pred_pos != answer_pos - 1 or pred_pos < 0 or answer_pos >= r["token_len"]:
                bad.append({"instance_id": r["instance_id"], "query": q,
                            "error": "bad answer_pos convention"})
                continue
            pred_decoded = alpha.tokenizer.decode(
                [int(mm[base + pred_pos])], clean_up_tokenization_spaces=False).strip()
            answer_decoded = alpha.tokenizer.decode(
                [int(mm[base + answer_pos])], clean_up_tokenization_spaces=False).strip()
            if pred_decoded != expected_pred or answer_decoded != answer_word:
                bad.append({"instance_id": r["instance_id"], "family": r.get("family"),
                            "pred_pos": pred_pos, "pred_decoded": pred_decoded,
                            "expected_pred": expected_pred, "answer_pos": answer_pos,
                            "answer_decoded": answer_decoded, "expected_answer": answer_word})
            if len(bad) >= 20:
                return {"name": "query_position_contract", "ok": False,
                        "checked": checked, "bad": bad}
    return {"name": "query_position_contract", "ok": checked > 0 and not bad,
            "checked": checked, "bad": bad}


def offset_contract(out_dir, split, records, alpha, seq_len=4096):
    if alpha is None:
        return {"name": f"offset_contract_{split}", "ok": False, "error": "alpha required"}
    bad = 0
    total = 0
    for r in records:
        path = os.path.join(out_dir, r["shard"])
        mm = np.memmap(path, dtype=np.uint16, mode="r")
        base = r["window_index"] * seq_len + r["start_offset"]
        for lab in r.get("label_positions", []):
            total += 1
            pos = base + lab["pos"]
            decoded = alpha.tokenizer.decode([int(mm[pos])], clean_up_tokenization_spaces=False)
            if decoded.strip() != lab["expected"]:
                bad += 1
    return {"name": f"offset_contract_{split}", "ok": bad == 0 and total > 0,
            "labels": total, "bad": bad}


def f3_age_audit(records):
    ages = []
    for r in records:
        if r.get("family") != "F3":
            continue
        for q in r.get("queries", []):
            age = q.get("age_write_to_query")
            if age is not None:
                ages.append(int(age))
    bins = {"<=32": 0, "33-128": 0, "129-512": 0, "513-1000": 0, ">1000": 0}
    for age in ages:
        if age <= 32:
            bins["<=32"] += 1
        elif age <= 128:
            bins["33-128"] += 1
        elif age <= 512:
            bins["129-512"] += 1
        elif age <= 1000:
            bins["513-1000"] += 1
        else:
            bins[">1000"] += 1
    return {"name": "f3_age", "ok": bool(ages) and bins[">1000"] > 0,
            "count": len(ages), "max": max(ages) if ages else None, "hist": bins}


def filler_scaffolding_audit(out_dir, records, alpha, seq_len=4096):
    if alpha is None:
        return {"name": "filler_scaffolding", "ok": False, "error": "alpha required"}
    filler_counts = defaultdict(Counter)
    structural_counts = defaultdict(Counter)
    for r in records:
        path = os.path.join(out_dir, r["shard"])
        mm = np.memmap(path, dtype=np.uint16, mode="r")
        base = r["window_index"] * seq_len + r["start_offset"]
        filler_positions = set()
        for start, length in r.get("filler_rle", []):
            filler_positions.update(range(start, start + length))
        for pos in range(r["token_len"]):
            word = alpha.tokenizer.decode([int(mm[base + pos])], clean_up_tokenization_spaces=False).strip()
            if word not in STRUCTURAL_TOKENS:
                continue
            if pos in filler_positions:
                filler_counts[r["family"]][word] += 1
            else:
                structural_counts[r["family"]][word] += 1
    bad = {fam: dict(c) for fam, c in filler_counts.items() if sum(c.values())}
    return {"name": "filler_scaffolding", "ok": not bad, "filler_counts": bad,
            "structural_counts": {fam: dict(c) for fam, c in structural_counts.items()}}


def f1_rule_consistency(records):
    bad = []
    checked = 0
    for r in records:
        if r.get("family") != "F1":
            continue
        event_pos = r["events"][0]["pos"]
        rid_a = r["rule_ids"]["A"]
        rid_b = r["rule_ids"]["B"]
        base_map = {w["key"]: w["value"] for w in r.get("writes", [])}
        for q in r.get("queries", []):
            checked += 1
            rule_at_q = None
            for start, rid, length in r.get("active_rule_rle", []):
                if start <= q["pos"] < start + length:
                    rule_at_q = rid
                    break
            expected_rule = rid_a if q["pos"] < event_pos else rid_b
            if rule_at_q != expected_rule:
                bad.append({"instance_id": r["instance_id"], "query_pos": q["pos"],
                            "rule_at_q": rule_at_q, "expected": expected_rule})
            if q["pos"] < event_pos and base_map.get(q["key"]) != q["answer"]:
                bad.append({"instance_id": r["instance_id"], "query_pos": q["pos"],
                            "answer": q["answer"], "base_answer": base_map.get(q["key"])})
            if q["pos"] >= event_pos and base_map.get(q["key"]) == q["answer"]:
                bad.append({"instance_id": r["instance_id"], "query_pos": q["pos"],
                            "answer": q["answer"], "error": "post-event answer equals base mapping"})
    return {"name": "f1_rule_consistency", "ok": not bad and checked > 0,
            "checked": checked, "bad": bad[:20]}


def run_qa(out_dir, alpha=None, seq_len=4096):
    checks = []
    if alpha is not None:
        checks.append(tokenizer_audit(alpha))
        checks.append(manifest_tokenizer_audit(out_dir, alpha))
    checks.append(packing_policy_audit(out_dir, seq_len))
    all_records = []
    for split in ("train", "eval"):
        path = os.path.join(out_dir, "sidecars", f"{split}.jsonl.gz")
        if os.path.exists(path):
            recs = _records(out_dir, split)
            all_records.extend(recs)
            checks.append(window_integrity(out_dir, split, recs, seq_len))
            checks.append(task_density_audit(out_dir, split, recs, seq_len))
            if alpha is not None:
                checks.append(offset_contract(out_dir, split, recs, alpha, seq_len))
                checks.append(packed_window_roundtrip(out_dir, split, alpha, seq_len))
                checks.append(window_background_scaffolding_audit(out_dir, split, recs, alpha, seq_len))
    checks += [distribution_audit(all_records), marker_audit(all_records),
               replay_audit(all_records), label_audit(all_records),
               query_position_contract(out_dir, all_records, alpha, seq_len),
               f3_age_audit(all_records),
               filler_scaffolding_audit(out_dir, all_records, alpha, seq_len),
               f1_rule_consistency(all_records)]
    ok = all(c["ok"] for c in checks)
    report = {"ok": ok, "checks": checks}
    with open(os.path.join(out_dir, "qa_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
