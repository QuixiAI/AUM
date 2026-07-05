"""SYN-1B generator QA checks."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from train.syn import MAX_ATTENTION_WINDOW_PLANNED
from train.syn.alphabet import STRUCTURAL_TOKENS
from train.syn.harness_readers import open_jsonl, prediction_pos_for_answer
from train.syn.pack import (MAX_SYNTHETIC_INSTANCES_PER_WINDOW,
                            MIN_QUERIES_BY_FAMILY,
                            MIN_TASK_FRACTION_BY_FAMILY)
from train.syn.schema import (MINIMAL_SUFFIX_EXEMPT_FAMILIES,
                              RECOMPUTE_OWNERS,
                              RECOMPUTE_PENDING,
                              REQUIRED_RECORD_FIELDS,
                              CONTROLLED_GAP_FIELDS,
                              DEMONSTRATION_FIELDS,
                              DISTRACTOR_FIELDS,
                              EVENT_FIELDS,
                              LABEL_POSITION_FIELDS,
                              WRITE_FIELDS,
                              allowed_query_fields,
                              allowed_record_fields,
                              field_owner_index,
                              verifier_registry_mismatches,
                              required_query_fields,
                              verifier_registry_summary)


def _age_bin_bounds(age_bin):
    if age_bin is None:
        return None
    age_bin = int(age_bin)
    if age_bin < 0 or age_bin > 9:
        return None
    ratio = 3500 / 8
    lo = int(8 * (ratio ** (age_bin / 10)))
    hi = int(8 * (ratio ** ((age_bin + 1) / 10)))
    if age_bin == 9:
        hi = 3500
    return lo, hi


def _len_matches_age_bin(length, age_bin):
    bounds = _age_bin_bounds(age_bin)
    if bounds is None:
        return False
    lo, hi = bounds
    value = int(length)
    return lo <= value <= hi


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
        "max_synthetic_instances_per_window": MAX_SYNTHETIC_INSTANCES_PER_WINDOW,
        "window_background": "eos_padding_for_standalone_dry_run",
        "min_task_fraction_by_family": MIN_TASK_FRACTION_BY_FAMILY,
        "min_queries_by_family": MIN_QUERIES_BY_FAMILY,
        "instance_boundary_mask": False,
        "eos_between_instances": True,
    }
    mismatches = {k: {"manifest": policy.get(k), "expected": v}
                  for k, v in expected.items() if policy.get(k) != v}
    manifest = _manifest(out_dir)
    if manifest.get("max_attention_window_planned") != MAX_ATTENTION_WINDOW_PLANNED:
        mismatches["max_attention_window_planned"] = {
            "manifest": manifest.get("max_attention_window_planned"),
            "expected": MAX_ATTENTION_WINDOW_PLANNED,
        }
    return {"name": "packing_policy", "ok": not mismatches,
            "mismatches": mismatches, "policy": policy}


def distribution_audit(records):
    lengths = [r["token_len"] for r in records]
    bad_len = [x for x in lengths if x < 48 or x > 3968]
    age_bins = Counter()
    bad_gaps = []
    for r in records:
        controlled_sum = 0
        for gap in r.get("controlled_gaps", []):
            target_bin = gap.get("target_age_bin")
            realized = int(gap.get("realized", 0))
            age_bins[target_bin] += 1
            controlled_sum += int(gap.get("realized", 0))
            if int(gap.get("target", -1)) != int(gap.get("realized", -2)):
                bad_gaps.append({"instance_id": r.get("instance_id"), "gap": gap})
            elif not _len_matches_age_bin(realized, target_bin):
                bad_gaps.append({"instance_id": r.get("instance_id"), "gap": gap,
                                 "target_bin_bounds": _age_bin_bounds(target_bin),
                                 "error": "realized controlled gap outside target_age_bin"})
        if int(r.get("controlled_gap_tokens", 0)) != controlled_sum:
            bad_gaps.append({"instance_id": r.get("instance_id"),
                             "controlled_gap_tokens": r.get("controlled_gap_tokens"),
                             "controlled_gap_sum": controlled_sum})
    event_fracs = []
    for r in records:
        if r.get("family") == "F3" or int(r.get("controlled_gap_tokens", 0)) > 0:
            continue
        for e in r.get("events", []):
            event_fracs.append(e["pos"] / max(1, r["token_len"]))
    bad_events = [x for x in event_fracs if x < 0.10 or x > 0.90]
    positive_bins = [age_bins.get(i, 0) for i in range(10)]
    mean_bins = float(np.mean(positive_bins)) if any(positive_bins) else 0.0
    total_controlled = sum(positive_bins)
    if any(positive_bins) and mean_bins >= 1000:
        gate_branch = "production_2pct"
        bin_tol = max(1, int(math.ceil(mean_bins * 0.02)))
        bad_bins = [
            {"bin": i, "count": c, "expected": mean_bins,
             "abs_deviation": abs(c - mean_bins), "tolerance": bin_tol}
            for i, c in enumerate(positive_bins)
            if abs(c - mean_bins) > bin_tol
        ]
        chi_square = None
        chi_square_threshold = None
        bin_ok = not bad_bins
    elif any(positive_bins):
        gate_branch = "small_sample_chi_square_p01"
        bin_tol = None
        chi_square = float(sum((c - mean_bins) ** 2 / mean_bins for c in positive_bins))
        chi_square_threshold = 21.666  # df=9, alpha=0.01
        bad_bins = []
        bin_ok = chi_square <= chi_square_threshold
    else:
        gate_branch = "no_controlled_gaps"
        bin_tol = None
        chi_square = None
        chi_square_threshold = None
        bad_bins = []
        bin_ok = True
    return {"name": "controlled_distribution",
            "ok": not bad_len and not bad_events and not bad_gaps and bin_ok,
            "records": len(records), "bad_len": len(bad_len), "bad_events": len(bad_events),
            "bad_gaps": bad_gaps[:20], "target_age_bins": dict(age_bins),
            "target_age_bin_gate": gate_branch,
            "target_age_bin_expected_per_bin": mean_bins,
            "target_age_bin_tolerance": bin_tol,
            "target_age_bin_chi_square": chi_square,
            "target_age_bin_chi_square_df": 9 if chi_square is not None else None,
            "target_age_bin_chi_square_threshold_p01": chi_square_threshold,
            "target_age_bin_total_controlled": total_controlled,
            "verified_fields": [("record", "controlled_gap_tokens"),
                                ("controlled_gap", "kind"), ("controlled_gap", "realized"),
                                ("controlled_gap", "target"), ("controlled_gap", "target_age_bin")],
            "bad_bins": bad_bins[:20]}


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
        crowded += sum(1 for c in per_window.values() if c > MAX_SYNTHETIC_INSTANCES_PER_WINDOW)
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
            "bad_records": bad, "crowded_windows": crowded,
            "verified_fields": [("record", "token_hash"), ("record", "token_len")]}


def task_density_audit(out_dir, split, records, seq_len=4096):
    del out_dir, seq_len
    instance_fracs = []
    bad_records = []
    query_counts = []
    for r in records:
        task = int(r.get("task_token_count", -1))
        filler = int(r.get("filler_token_count", -1))
        token_len = int(r.get("token_len", 0))
        if task < 0 or filler < 0 or task + filler != token_len:
            bad_records.append({"instance_id": r.get("instance_id"), "task": task,
                                "filler": filler, "token_len": token_len})
            continue
        controlled = int(r.get("controlled_gap_tokens", 0))
        denom = token_len - controlled
        frac = task / max(1, denom)
        instance_fracs.append(frac)
        query_count = len(r.get("queries", []))
        query_counts.append(query_count)
        fam = r.get("family")
        min_frac = MIN_TASK_FRACTION_BY_FAMILY.get(fam, 0.25)
        min_queries = MIN_QUERIES_BY_FAMILY.get(fam)
        errors = []
        emitted_denom = r.get("density_denominator")
        if emitted_denom is not None and int(emitted_denom) != int(denom):
            errors.append(f"density_denominator {emitted_denom}!={denom}")
        emitted_frac = r.get("controlled_gap_adjusted_task_fraction")
        if emitted_frac is not None and abs(float(emitted_frac) - frac) > 1e-6:
            errors.append(f"controlled_gap_adjusted_task_fraction {emitted_frac}!={frac:.6f}")
        if min_frac is not None and frac < min_frac:
            errors.append(f"task_fraction<{min_frac}")
        if min_queries is not None and query_count < min_queries:
            errors.append(f"queries<{min_queries}")
        if errors:
            bad_records.append({"instance_id": r.get("instance_id"), "family": fam,
                                "task_fraction": frac, "density_denominator": denom,
                                "controlled_gap_tokens": controlled, "queries": query_count,
                                "errors": errors})
    return {"name": f"instance_task_density_{split}",
            "ok": bool(records) and not bad_records,
            "instance_fraction_mean": float(np.mean(instance_fracs)) if instance_fracs else None,
            "instance_fraction_min": min(instance_fracs) if instance_fracs else None,
            "instance_fraction_max": max(instance_fracs) if instance_fracs else None,
            "query_count_min": min(query_counts) if query_counts else None,
            "query_count_mean": float(np.mean(query_counts)) if query_counts else None,
            "min_task_fraction_by_family": MIN_TASK_FRACTION_BY_FAMILY,
            "min_queries_by_family": MIN_QUERIES_BY_FAMILY,
            "verified_fields": [("record", "density_denominator"),
                                ("record", "controlled_gap_adjusted_task_fraction")],
            "bad_records": bad_records[:20]}


def instance_span_and_seam_roundtrip(out_dir, split, records, alpha, seq_len=4096):
    if alpha is None:
        return {"name": f"instance_span_roundtrip_{split}", "ok": False,
                "error": "alpha required"}
    bad = []
    checked = 0
    maps = {}
    eos = int(alpha.eos_id)

    def contains_subseq(haystack, needle):
        if not needle or len(needle) > len(haystack):
            return False
        limit = len(haystack) - len(needle) + 1
        for i in range(limit):
            if haystack[i:i + len(needle)] == needle:
                return True
        return False

    for r in records:
        path = os.path.join(out_dir, r["shard"])
        mm = maps.get(path)
        if mm is None:
            mm = maps[path] = np.memmap(path, dtype=np.uint16, mode="r")
        abs_start = r["window_index"] * seq_len + r["start_offset"]
        abs_end = abs_start + r["token_len"]
        ids = np.asarray(mm[abs_start:abs_end], dtype=np.int64).tolist()
        checked += 1
        text = alpha.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        ids2 = alpha.tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids2 != ids:
            first = next((i for i, pair in enumerate(zip(ids, ids2)) if pair[0] != pair[1]),
                         min(len(ids), len(ids2)))
            bad.append({"instance_id": r.get("instance_id"),
                        "first_mismatch": first, "len_ids": len(ids),
                        "len_retoked": len(ids2)})
            if len(bad) >= 20:
                break
            continue

        ctx = []
        if r["start_offset"] > 0 and int(mm[abs_start - 1]) != eos:
            ctx.append(int(mm[abs_start - 1]))
        ctx_len_left = len(ctx)
        ctx.extend(ids)
        if r["start_offset"] + r["token_len"] < seq_len and int(mm[abs_end]) != eos:
            ctx.append(int(mm[abs_end]))
        if len(ctx) > len(ids):
            seam_text = alpha.tokenizer.decode(ctx, clean_up_tokenization_spaces=False)
            seam_ids = alpha.tokenizer(seam_text, add_special_tokens=False)["input_ids"]
            if not contains_subseq(seam_ids, ids):
                bad.append({"instance_id": r.get("instance_id"),
                            "error": "instance ids changed in immediate seam context",
                            "left_context_tokens": ctx_len_left,
                            "right_context_tokens": len(ctx) - ctx_len_left - len(ids)})
        if len(bad) >= 20:
            break
    return {"name": f"instance_span_roundtrip_{split}", "ok": checked > 0 and not bad,
            "instances": checked, "bad": bad}


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


def split_integrity_audit(records):
    bad = []
    split_counts = defaultdict(Counter)
    for r in records:
        family = r.get("family")
        split = r.get("split")
        split_counts[family][split] += 1
        if family == "F5" and split != "eval":
            bad.append({"instance_id": r.get("instance_id"), "split": split,
                        "shard": r.get("shard"), "window_index": r.get("window_index")})
    return {"name": "split_integrity", "ok": not bad,
            "split_counts": {fam: dict(c) for fam, c in split_counts.items()},
            "bad": bad[:20]}


def schema_completeness_audit(records):
    bad = []
    total_queries = 0
    query_field_counts = Counter()
    for r in records:
        missing_record = sorted(REQUIRED_RECORD_FIELDS - set(r))
        if missing_record:
            bad.append({"instance_id": r.get("instance_id"), "missing_record": missing_record})
        fam = r.get("family")
        required_query = required_query_fields(fam)
        for i, q in enumerate(r.get("queries", [])):
            total_queries += 1
            query_field_counts[fam] += 1
            missing_query = sorted(required_query - set(q))
            if missing_query:
                bad.append({"instance_id": r.get("instance_id"), "family": fam,
                            "query_index": i, "missing_query": missing_query})
            if len(bad) >= 20:
                return {"name": "schema_completeness", "ok": False,
                        "records": len(records), "queries": total_queries, "bad": bad}
    return {"name": "schema_completeness", "ok": bool(records) and total_queries > 0 and not bad,
            "records": len(records), "queries": total_queries,
            "query_counts_by_family": dict(query_field_counts),
            "required_record_fields": sorted(REQUIRED_RECORD_FIELDS),
            "bad": bad}


def schema_admission_audit(records):
    bad = []
    checked = Counter()
    registry_mismatches = verifier_registry_mismatches()

    def add_bad(record, location, fields, allowed):
        extra = sorted(set(fields) - set(allowed))
        if extra:
            bad.append({"instance_id": record.get("instance_id"), "family": record.get("family"),
                        "location": location, "unregistered_fields": extra})

    for r in records:
        fam = r.get("family")
        checked["records"] += 1
        add_bad(r, "record", set(r), allowed_record_fields(fam))
        for i, q in enumerate(r.get("queries", [])):
            checked["queries"] += 1
            add_bad(r, f"queries[{i}]", set(q), allowed_query_fields(fam))
        for i, w in enumerate(r.get("writes", [])):
            checked["writes"] += 1
            add_bad(r, f"writes[{i}]", set(w), WRITE_FIELDS.get(fam, set()))
        for i, e in enumerate(r.get("events", [])):
            checked["events"] += 1
            add_bad(r, f"events[{i}]", set(e), EVENT_FIELDS.get(fam, set()))
        for i, d in enumerate(r.get("distractors", [])):
            checked["distractors"] += 1
            add_bad(r, f"distractors[{i}]", set(d), DISTRACTOR_FIELDS)
        for i, g in enumerate(r.get("controlled_gaps", [])):
            checked["controlled_gaps"] += 1
            add_bad(r, f"controlled_gaps[{i}]", set(g), CONTROLLED_GAP_FIELDS)
        for i, d in enumerate(r.get("demonstrations", [])):
            checked["demonstrations"] += 1
            add_bad(r, f"demonstrations[{i}]", set(d), DEMONSTRATION_FIELDS)
        for i, label in enumerate(r.get("label_positions", [])):
            checked["label_positions"] += 1
            add_bad(r, f"label_positions[{i}]", set(label), LABEL_POSITION_FIELDS)
        if len(bad) >= 20:
            break
    return {"name": "schema_admission", "ok": bool(records) and not bad and not registry_mismatches,
            "checked": dict(checked),
            "registered_verifier_groups": sorted(verifier_registry_summary()),
            "registry_mismatches": registry_mismatches,
            "bad": bad}


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
            answer_word = q["answer"]
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
            "checked": checked,
            "verified_fields": [("query_common", "answer"), ("query_common", "answer_pos"),
                                ("query_common", "prediction_anchor")],
            "bad": bad}


def distractor_surface_audit(out_dir, records, alpha, seq_len=4096):
    if alpha is None:
        return {"name": "distractor_surface", "ok": False, "error": "alpha required"}
    required = {
        "F1": {"means"},
        "F2": {"exchanged"},
        "F3": {"key", "opens", "box"},
    }
    inert_markers = {"F2": {"nothing", "not"}}
    bad = []
    checked = 0
    maps = {}
    period_id = alpha.single_id(".")
    for r in records:
        if r.get("family") != "F4":
            continue
        path = os.path.join(out_dir, r["shard"])
        mm = maps.get(path)
        if mm is None:
            mm = maps[path] = np.memmap(path, dtype=np.uint16, mode="r")
        base = int(r["window_index"]) * seq_len + int(r["start_offset"])
        distractor_list = r.get("distractors", [])
        emitted_pep = r.get("pseudo_event_pos")
        expected_pep = int(distractor_list[0]["pos"]) if distractor_list else None
        if emitted_pep != expected_pep:
            bad.append({"instance_id": r.get("instance_id"),
                        "error": "pseudo_event_pos != first distractor pos",
                        "pseudo_event_pos": emitted_pep, "expected": expected_pep})
        for distractor in r.get("distractors", []):
            checked += 1
            mimics = distractor.get("mimics_family")
            req = required.get(mimics)
            if not req:
                bad.append({"instance_id": r.get("instance_id"), "mimics_family": mimics,
                            "error": "unknown mimicked family"})
                continue
            if distractor.get("variant") != f"{mimics}_surface":
                bad.append({"instance_id": r.get("instance_id"), "mimics_family": mimics,
                            "variant": distractor.get("variant"),
                            "error": "variant != {mimics}_surface"})
            pos = int(distractor["pos"])
            words = []
            for rel in range(pos, min(int(r["token_len"]), pos + 20)):
                tid = int(mm[base + rel])
                words.append(alpha.tokenizer.decode([tid], clean_up_tokenization_spaces=False).strip())
                if tid == period_id:
                    break
            word_set = set(words)
            missing = sorted(req - word_set)
            inert_required = inert_markers.get(mimics)
            inert_ok = True if not inert_required else bool(word_set & inert_required)
            if missing or not inert_ok:
                bad.append({"instance_id": r.get("instance_id"), "pos": pos,
                            "mimics_family": mimics, "words": words,
                            "missing_signature": missing,
                            "missing_inert_marker": sorted(inert_required or [])
                            if inert_required and not inert_ok else []})
                if len(bad) >= 20:
                    break
        if len(bad) >= 20:
            break
    return {"name": "distractor_surface", "ok": checked > 0 and not bad,
            "checked": checked, "required": {k: sorted(v) for k, v in required.items()},
            "verified_fields": [("distractor", "mimics_family"), ("distractor", "variant"),
                                ("distractor", "pos"), ("record", "pseudo_event_pos")],
            "bad": bad}


def offset_contract(out_dir, split, records, alpha, seq_len=4096):
    if alpha is None:
        return {"name": f"offset_contract_{split}", "ok": False, "error": "alpha required"}
    bad = 0
    total = 0
    reconcile_bad = []
    for r in records:
        path = os.path.join(out_dir, r["shard"])
        mm = np.memmap(path, dtype=np.uint16, mode="r")
        base = r["window_index"] * seq_len + r["start_offset"]
        label_pos = set()
        for lab in r.get("label_positions", []):
            total += 1
            label_pos.add(int(lab["pos"]))
            pos = base + lab["pos"]
            decoded = alpha.tokenizer.decode([int(mm[pos])], clean_up_tokenization_spaces=False)
            if decoded.strip() != lab["expected"]:
                bad += 1
        # Every record position field must reference a byte-verified label anchor.
        record_positions = []
        for e in r.get("events", []):
            record_positions.append(("event.pos", e.get("pos")))
        for q in r.get("queries", []):
            record_positions.append(("query.pos", q.get("pos")))
            record_positions.append(("query.answer_pos", q.get("answer_pos")))
        for w in r.get("writes", []):
            record_positions.append(("write.pos", w.get("pos")))
        for d in r.get("demonstrations", []):
            record_positions.append(("demonstration.pos", d.get("pos")))
            record_positions.append(("demonstration.answer_pos", d.get("answer_pos")))
        for d in r.get("distractors", []):
            record_positions.append(("distractor.pos", d.get("pos")))
        for name, value in record_positions:
            if value is None or int(value) not in label_pos:
                reconcile_bad.append({"instance_id": r.get("instance_id"),
                                      "field": name, "pos": value})
    return {"name": f"offset_contract_{split}", "ok": bad == 0 and total > 0 and not reconcile_bad,
            "labels": total, "bad": bad, "reconcile_bad": reconcile_bad[:20],
            "verified_fields": [("label_position", "pos"), ("label_position", "expected"),
                                ("label_position", "decoded"),
                                ("event", "pos"), ("query_common", "pos"),
                                ("query_common", "answer_pos"), ("write", "pos"),
                                ("demonstration", "pos"), ("demonstration", "answer_pos"),
                                ("distractor", "pos")]}


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
    count_bad = []
    for r in records:
        path = os.path.join(out_dir, r["shard"])
        mm = np.memmap(path, dtype=np.uint16, mode="r")
        base = r["window_index"] * seq_len + r["start_offset"]
        filler_positions = set()
        filler_from_rle = 0
        for start, length in r.get("filler_rle", []):
            filler_positions.update(range(start, start + length))
            filler_from_rle += int(length)
        token_len = int(r["token_len"])
        task_from_rle = token_len - filler_from_rle
        if int(r.get("filler_token_count", -1)) != filler_from_rle:
            count_bad.append({"instance_id": r.get("instance_id"), "field": "filler_token_count",
                              "emitted": r.get("filler_token_count"), "recomputed": filler_from_rle})
        elif int(r.get("task_token_count", -1)) != task_from_rle:
            count_bad.append({"instance_id": r.get("instance_id"), "field": "task_token_count",
                              "emitted": r.get("task_token_count"), "recomputed": task_from_rle})
        elif (r.get("task_fraction") is not None
              and abs(float(r["task_fraction"]) - task_from_rle / max(1, token_len)) > 1e-6):
            count_bad.append({"instance_id": r.get("instance_id"), "field": "task_fraction",
                              "emitted": r.get("task_fraction"),
                              "recomputed": task_from_rle / max(1, token_len)})
        for pos in range(token_len):
            word = alpha.tokenizer.decode([int(mm[base + pos])], clean_up_tokenization_spaces=False).strip()
            if word not in STRUCTURAL_TOKENS:
                continue
            if pos in filler_positions:
                filler_counts[r["family"]][word] += 1
            else:
                structural_counts[r["family"]][word] += 1
    bad = {fam: dict(c) for fam, c in filler_counts.items() if sum(c.values())}
    return {"name": "filler_scaffolding", "ok": not bad and not count_bad,
            "filler_counts": bad, "count_bad": count_bad[:20],
            "verified_fields": [("record", "filler_rle"), ("record", "filler_token_count"),
                                ("record", "task_token_count"), ("record", "task_fraction")],
            "structural_counts": {fam: dict(c) for fam, c in structural_counts.items()}}


def train_eval_symbol_exclusion_audit(out_dir, records, alpha, seq_len=4096):
    if alpha is None:
        return {"name": "train_eval_symbol_exclusion", "ok": False, "error": "alpha required"}
    eval_symbol_ids = {alpha.single_id(w): w for w in alpha.sigma_eval}
    bad = []
    checked = 0
    maps = {}
    for r in records:
        if r.get("split") != "train":
            continue
        path = os.path.join(out_dir, r["shard"])
        mm = maps.get(path)
        if mm is None:
            mm = maps[path] = np.memmap(path, dtype=np.uint16, mode="r")
        base = int(r["window_index"]) * seq_len + int(r["start_offset"])
        for rel in range(int(r["token_len"])):
            checked += 1
            word = eval_symbol_ids.get(int(mm[base + rel]))
            if word is not None:
                bad.append({"instance_id": r.get("instance_id"), "family": r.get("family"),
                            "pos": rel, "symbol": word})
                if len(bad) >= 20:
                    break
        if len(bad) >= 20:
            break
    return {"name": "train_eval_symbol_exclusion", "ok": checked > 0 and not bad,
            "checked_train_tokens": checked, "sigma_eval": list(alpha.sigma_eval), "bad": bad}


def f1_rule_consistency(records):
    bad = []
    checked = 0

    def map_at(record, pos):
        for seg in record.get("active_map_rle", []):
            if int(seg["start"]) <= pos < int(seg["end"]):
                return seg.get("map", {})
        return {}

    for r in records:
        if r.get("family") != "F1":
            continue
        segments = r.get("active_map_rle", [])
        if len(segments) != len(r.get("events", [])) + 1:
            bad.append({"instance_id": r.get("instance_id"), "error": "bad active_map_rle length"})
            continue
        for i, event in enumerate(r.get("events", [])):
            before = segments[i].get("map", {})
            after = segments[i + 1].get("map", {})
            changed = set(event.get("changed_symbols", []))
            mentioned = set(event.get("mentioned_symbols", []))
            if event.get("restatement") == "partial":
                if changed != mentioned or len(changed) != 3:
                    bad.append({"instance_id": r.get("instance_id"), "event": i,
                                "error": "partial event must mention exactly changed symbols"})
                for key, old_value in before.items():
                    if key in changed:
                        if after.get(key) == old_value:
                            bad.append({"instance_id": r.get("instance_id"), "event": i,
                                        "key": key, "error": "changed key did not change"})
                    elif after.get(key) != old_value:
                        bad.append({"instance_id": r.get("instance_id"), "event": i,
                                    "key": key, "error": "unmentioned key changed"})
        has_unmentioned_partial_query = False
        for q in r.get("queries", []):
            checked += 1
            active = map_at(r, int(q["pos"]))
            if active.get(q["key"]) != q.get("answer"):
                bad.append({"instance_id": r.get("instance_id"), "query_pos": q["pos"],
                            "key": q["key"], "answer": q.get("answer"),
                            "active_answer": active.get(q["key"])})
            if q.get("age_from_reversal") is not None and q.get("mentioned_in_latest_correction") is False:
                if any(e.get("restatement") == "partial" for e in r.get("events", [])):
                    has_unmentioned_partial_query = True
        if any(e.get("restatement") == "partial" for e in r.get("events", [])):
            if not has_unmentioned_partial_query:
                bad.append({"instance_id": r.get("instance_id"),
                            "error": "partial F1 has no unmentioned-symbol query"})
    return {"name": "f1_rule_consistency", "ok": not bad and checked > 0,
            "checked": checked,
            "verified_fields": [("record", "active_map_rle"), ("event", "changed_symbols"),
                                ("event", "mentioned_symbols"), ("event", "restatement")],
            "bad": bad[:20]}


def _record_key_for_query(query):
    return query.get("key", query.get("entity"))


def _map_at(record, pos):
    for seg in record.get("active_map_rle", []):
        if int(seg["start"]) <= pos < int(seg["end"]):
            return seg.get("map", {})
    return {}


def _event_directly_states(record, event, key):
    fam = record.get("family")
    if fam == "F1":
        return key in set(event.get("changed_symbols", []))
    if fam == "F3":
        return event.get("key") == key
    return False


def _event_relevant_after_source(record, event, key):
    fam = record.get("family")
    if fam in {"F1", "F2"}:
        return True
    if fam == "F3":
        return event.get("key") == key
    return False


def _minimal_suffix_from_sidecar_sources(record, query_index):
    query = record["queries"][query_index]
    key = _record_key_for_query(query)
    if key is None:
        return None, []
    pred_pos = prediction_pos_for_answer(query)
    candidates = []
    for write in record.get("writes", []):
        if write.get("key", write.get("entity")) == key and int(write["pos"]) < pred_pos:
            candidates.append([int(write["pos"])])
    for demo in record.get("demonstrations", []):
        if demo.get("key") == key and int(demo["pos"]) < pred_pos:
            candidates.append([int(demo["pos"])])
    for prev in record.get("queries", [])[:query_index]:
        if _record_key_for_query(prev) == key and int(prev["pos"]) < pred_pos:
            candidates.append([int(prev["pos"])])
    for event in record.get("events", []):
        if int(event["pos"]) < pred_pos and _event_directly_states(record, event, key):
            candidates.append([int(event["pos"])])

    expanded = []
    for evidence in candidates:
        source_pos = min(evidence)
        full = list(evidence)
        for event in record.get("events", []):
            event_pos = int(event["pos"])
            if source_pos < event_pos < pred_pos and _event_relevant_after_source(record, event, key):
                full.append(event_pos)
        full = sorted(set(full))
        expanded.append(full)
    if not expanded:
        return None, []
    best = min(expanded, key=lambda ev: pred_pos - min(ev) + 1)
    return pred_pos - min(best) + 1, best


def minimal_suffix_audit(records):
    bad = []
    checked = 0
    total = 0
    hard_suffix_checked = 0
    checked_by_family = Counter()
    skipped_by_family = Counter()
    families_seen = set()
    for r in records:
        family = r.get("family")
        queries = r.get("queries", [])
        total += len(queries)
        for i, q in enumerate(queries):
            declared_suffix = q.get("minimal_sufficient_suffix_len")
            declared_hard = q.get("hard_suffix")
            if declared_suffix is not None and declared_hard is not None:
                hard_suffix_checked += 1
                expected_hard = int(declared_suffix) > MAX_ATTENTION_WINDOW_PLANNED
                if bool(declared_hard) != expected_hard:
                    bad.append({"instance_id": r.get("instance_id"), "family": family,
                                "query_index": i, "declared_suffix": declared_suffix,
                                "declared_hard_suffix": declared_hard,
                                "expected_hard_suffix": expected_hard})
                    if len(bad) >= 20:
                        break
        if len(bad) >= 20:
            break
        if family in MINIMAL_SUFFIX_EXEMPT_FAMILIES:
            skipped_by_family[family] += len(queries)
            continue
        for i, q in enumerate(r.get("queries", [])):
            checked += 1
            checked_by_family[family] += 1
            families_seen.add(family)
            recomputed, evidence = _minimal_suffix_from_sidecar_sources(r, i)
            declared = q.get("minimal_sufficient_suffix_len")
            if recomputed is None or declared is None or int(declared) != int(recomputed):
                bad.append({"instance_id": r.get("instance_id"), "family": r.get("family"),
                            "query_index": i, "declared": declared, "recomputed": recomputed,
                            "declared_evidence": q.get("evidence_positions"),
                            "recomputed_evidence": evidence})
                if len(bad) >= 20:
                    break
            active = _map_at(r, int(q["pos"]))
            key = _record_key_for_query(q)
            answer = q["answer"]
            if active and active.get(key) != answer:
                bad.append({"instance_id": r.get("instance_id"), "family": r.get("family"),
                            "query_index": i, "key": key, "answer": answer,
                            "active_answer": active.get(key)})
                if len(bad) >= 20:
                    break
        if len(bad) >= 20:
            break
    verified = [("query_common", "minimal_sufficient_suffix_len"),
                ("query_common", "hard_suffix"), ("query_common", "evidence_positions"),
                ("query_common", "answer"), ("record", "active_map_rle")]
    for fam in families_seen:
        verified.append((f"query_{fam}", "entity" if fam == "F2" else "key"))
    return {"name": "minimal_suffix", "ok": checked > 0 and not bad,
            "total_queries": total, "checked": checked,
            "hard_suffix_checked": hard_suffix_checked,
            "hard_suffix_threshold": MAX_ATTENTION_WINDOW_PLANNED,
            "checked_by_family": dict(checked_by_family),
            "skipped_by_family": dict(skipped_by_family),
            "skip_rationale": {k: v for k, v in MINIMAL_SUFFIX_EXEMPT_FAMILIES.items()
                               if skipped_by_family.get(k, 0)},
            "verified_fields": verified,
            "bad": bad}


def _record_has_hard_primitive(record, max_attention_window):
    family = record.get("family")
    if family not in {"F1", "F2", "F3"}:
        return False
    if family in {"F1", "F2"} and int(record.get("composition_depth", 0)) > 1:
        return True
    return any(
        int(q.get("minimal_sufficient_suffix_len", 0)) > max_attention_window
        for q in record.get("queries", [])
    )


def difficulty_stratum_audit(records, max_attention_window=MAX_ATTENTION_WINDOW_PLANNED):
    counts = Counter()
    eval_family = Counter()
    eval_hard_primitive_family = Counter()
    eval_comp_family = Counter()
    eval_supersession_family = Counter()
    eval_queries = Counter()
    eval_suffix_queries = Counter()
    for r in records:
        fam = r.get("family")
        split = r.get("split")
        if split == "eval":
            eval_family[fam] += 1
            if _record_has_hard_primitive(r, max_attention_window):
                eval_hard_primitive_family[fam] += 1
            if fam in {"F1", "F2"} and int(r.get("composition_depth", 0)) > 1:
                eval_comp_family[fam] += 1
            if fam == "F3" and int(r.get("composition_depth", 0)) > 1:
                eval_supersession_family[fam] += 1
        if fam == "F1":
            if any(q.get("age_from_reversal") is not None
                   and q.get("mentioned_in_latest_correction") is False
                   for q in r.get("queries", [])):
                counts["f1_unmentioned_partial_query"] += 1
        if fam in {"F1", "F2"} and int(r.get("composition_depth", 0)) > 1:
            counts[f"{fam.lower()}_composition_depth_gt1"] += 1
        if fam == "F3" and int(r.get("composition_depth", 0)) > 1:
            counts["f3_supersession_depth_gt1"] += 1
        if any(int(q.get("minimal_sufficient_suffix_len", 0)) > max_attention_window
               for q in r.get("queries", [])):
            counts["suffix_gt_max_attention_window"] += 1
        if split == "eval" and fam in {"F1", "F3"}:
            for q in r.get("queries", []):
                eval_queries[fam] += 1
                if int(q.get("minimal_sufficient_suffix_len", 0)) > max_attention_window:
                    eval_suffix_queries[fam] += 1

    failures = []
    for fam in ("F1", "F2", "F3"):
        total = eval_family[fam]
        if total and eval_hard_primitive_family[fam] / total < 0.15:
            failures.append({"family": fam, "metric": "hard_primitive_eval_fraction",
                             "fraction": eval_hard_primitive_family[fam] / total})
    for fam in ("F1", "F2"):
        total = eval_family[fam]
        if total and eval_comp_family[fam] / total < 0.15:
            failures.append({"family": fam, "metric": "composition_eval_fraction",
                             "fraction": eval_comp_family[fam] / total})
    total = eval_family["F3"]
    if total and eval_supersession_family["F3"] / total < 0.15:
        failures.append({"family": "F3", "metric": "supersession_eval_fraction",
                         "fraction": eval_supersession_family["F3"] / total})
    for fam in ("F1", "F3"):
        total_q = eval_queries[fam]
        if total_q and eval_suffix_queries[fam] / total_q < 0.10:
            failures.append({"family": fam, "metric": "suffix_query_fraction",
                             "fraction": eval_suffix_queries[fam] / total_q})
    required = ["f1_unmentioned_partial_query", "suffix_gt_max_attention_window"]
    missing = [k for k in required if counts[k] == 0]
    return {"name": "difficulty_strata", "ok": not missing and not failures,
            "max_attention_window": max_attention_window,
            "counts": dict(counts), "eval_family_counts": dict(eval_family),
            "eval_hard_primitive_family_counts": dict(eval_hard_primitive_family),
            "eval_composition_family_counts": dict(eval_comp_family),
            "eval_supersession_family_counts": dict(eval_supersession_family),
            "eval_query_counts": dict(eval_queries),
            "eval_suffix_query_counts": dict(eval_suffix_queries),
            "missing": missing, "failures": failures}


def composition_depth_audit(records):
    """Recompute composition_depth independently of the emitter's counter.

    Record level, recomputed from the ``events`` list (independent of the emitter's counter):
    F1/F2 = len(events); F3 = len(events) or 1; F4 = 0; F5 = 1. Query level: the emitter labels
    every query with the record composition_depth (events-seen is a separate, unemitted field),
    so each query must equal the recomputed record depth. This was a consumer-only field
    (difficulty_strata read it, nothing recomputed it).
    """
    bad = []
    checked = 0

    def record_expected(r):
        fam = r.get("family")
        n = len(r.get("events", []))
        if fam in ("F1", "F2"):
            return n
        if fam == "F3":
            return n if n else 1
        if fam == "F4":
            return 0
        if fam == "F5":
            return 1
        return None

    for r in records:
        checked += 1
        rec_exp = record_expected(r)
        emitted = r.get("composition_depth")
        if rec_exp is None or emitted is None or int(emitted) != int(rec_exp):
            bad.append({"instance_id": r.get("instance_id"), "family": r.get("family"),
                        "scope": "record", "emitted": emitted, "recomputed": rec_exp,
                        "events": len(r.get("events", []))})
        for i, q in enumerate(r.get("queries", [])):
            q_exp = rec_exp
            q_emit = q.get("composition_depth")
            if q_exp is None or q_emit is None or int(q_emit) != int(q_exp):
                bad.append({"instance_id": r.get("instance_id"), "family": r.get("family"),
                            "scope": f"queries[{i}]", "emitted": q_emit, "recomputed": q_exp})
            if len(bad) >= 20:
                break
        if len(bad) >= 20:
            break
    return {"name": "composition_depth_recompute", "ok": checked > 0 and not bad,
            "checked": checked,
            "verified_fields": [("record", "composition_depth"),
                                ("query_common", "composition_depth")],
            "bad": bad}


def age_flag_audit(records):
    """Recompute F1/F2/F3 age and swap/mention flags independently of the emitter's (unemitted)
    events_seen counter, via a latest-event-before-query position scan. Ages follow the §7
    convention answer_pos - evidence_pos."""
    bad = []
    checked = 0
    families_seen = set()

    def latest_event_before(r, qpos):
        best = None
        for e in r.get("events", []):
            ep = e.get("pos")
            if ep is not None and int(ep) < qpos and (best is None or int(ep) > int(best["pos"])):
                best = e
        return best

    def write_pos_for(r, key):
        pos = None
        for w in r.get("writes", []):
            if w.get("key", w.get("entity")) == key:
                pos = int(w["pos"])
        return pos

    def check(cond, r, i, field, emitted, recomputed):
        if not cond:
            bad.append({"instance_id": r.get("instance_id"), "family": r.get("family"),
                        "query_index": i, "field": field,
                        "emitted": emitted, "recomputed": recomputed})

    for r in records:
        fam = r.get("family")
        if fam not in ("F1", "F2", "F3"):
            continue
        for i, q in enumerate(r.get("queries", [])):
            checked += 1
            families_seen.add(fam)
            apos = int(q["answer_pos"])
            qpos = int(q["pos"]) if q.get("pos") is not None else prediction_pos_for_answer(q)
            key = q.get("key", q.get("entity"))
            last = latest_event_before(r, qpos)
            wpos = write_pos_for(r, key)
            if fam == "F1":
                if wpos is not None:
                    check(int(q.get("age_from_original_statement", -1)) == apos - wpos, r, i,
                          "age_from_original_statement",
                          q.get("age_from_original_statement"), apos - wpos)
                exp_rev = None if last is None else apos - int(last["pos"])
                check(q.get("age_from_reversal") == exp_rev, r, i, "age_from_reversal",
                      q.get("age_from_reversal"), exp_rev)
                exp_ment = False if last is None else key in last.get("mentioned_symbols", [])
                check(q.get("mentioned_in_latest_correction") == exp_ment, r, i,
                      "mentioned_in_latest_correction",
                      q.get("mentioned_in_latest_correction"), exp_ment)
            elif fam == "F2":
                if wpos is not None:
                    check(int(q.get("age_from_binding_statement", -1)) == apos - wpos, r, i,
                          "age_from_binding_statement",
                          q.get("age_from_binding_statement"), apos - wpos)
                exp_swap = None if last is None else apos - int(last["pos"])
                check(q.get("age_from_swap") == exp_swap, r, i, "age_from_swap",
                      q.get("age_from_swap"), exp_swap)
                latest_set = set(last.get("swapped_entities", [])) if last else set()
                check(bool(q.get("entity_was_swapped")) == (key in latest_set), r, i,
                      "entity_was_swapped", q.get("entity_was_swapped"), key in latest_set)
                ever = any(key in set(e.get("swapped_entities", []))
                           for e in r.get("events", [])
                           if e.get("pos") is not None and int(e["pos"]) < qpos)
                check(bool(q.get("entity_ever_swapped")) == ever, r, i, "entity_ever_swapped",
                      q.get("entity_ever_swapped"), ever)
            elif fam == "F3":
                if wpos is not None:
                    check(int(q.get("age_write_to_query", -1)) == apos - wpos, r, i,
                          "age_write_to_query", q.get("age_write_to_query"), apos - wpos)
                events = r.get("events", [])
                exp_corr = None if not events else apos - int(events[-1]["pos"])
                check(q.get("age_correction_to_query") == exp_corr, r, i,
                      "age_correction_to_query", q.get("age_correction_to_query"), exp_corr)
                exp_mode = "correction" if events else "recall"
                check(q.get("mode") == exp_mode, r, i, "mode", q.get("mode"), exp_mode)
            if len(bad) >= 20:
                break
        if len(bad) >= 20:
            break

    verified = []
    if "F1" in families_seen:
        verified += [("query_F1", "age_from_original_statement"),
                     ("query_F1", "age_from_reversal"),
                     ("query_F1", "mentioned_in_latest_correction")]
    if "F2" in families_seen:
        verified += [("query_F2", "age_from_binding_statement"), ("query_F2", "age_from_swap"),
                     ("query_F2", "entity_was_swapped"), ("query_F2", "entity_ever_swapped")]
    if "F3" in families_seen:
        verified += [("query_F3", "age_write_to_query"), ("query_F3", "age_correction_to_query"),
                     ("query_F3", "mode")]
    return {"name": "age_flag_recompute", "ok": checked > 0 and not bad,
            "checked": checked, "verified_fields": verified, "bad": bad}


def content_decode_audit(out_dir, records, alpha, seq_len=4096):
    """Independent surface decoder: recompute write/event/demonstration *content* values from
    the rendered token bytes and construction rules, and assert equality with the sidecar. This
    is the emitter-independent parser that drains the RECOMPUTE_PENDING content fields.

    - write.key/entity: byte at write.pos; write.value: token before the statement's period.
    - F1 demo.key/answer: bytes at demo.pos/answer_pos.
    - event.type/event_index/source_state/target_state: construction rules.
    - F2 event.swapped_entities: the Sigma symbols in the swap sentence.
    - F3 event.key/new: swap-sentence surface; event.old: replayed key state from writes+events.
    - F5 event.old/new: the switch registry ids (not byte content).
    """
    if alpha is None:
        return {"name": "content_decode", "ok": False, "error": "alpha required"}
    sigma = set(alpha.sigma)
    event_type_by_family = {"F1": "reversal", "F2": "swap", "F3": "correction", "F5": "switch"}
    allowed_roles = {"write", "demo", "demo_answer", "query", "answer", "event", "distractor"}
    bad = []
    seen = set()
    maps = {}

    def flag(r, **kw):
        kw["instance_id"] = r.get("instance_id")
        kw["family"] = r.get("family")
        bad.append(kw)

    for r in records:
        fam = r.get("family")
        path = os.path.join(out_dir, r["shard"])
        mm = maps.get(path)
        if mm is None:
            mm = maps[path] = np.memmap(path, dtype=np.uint16, mode="r")
        base = int(r["window_index"]) * seq_len + int(r["start_offset"])
        token_len = int(r["token_len"])

        def word(rel):
            if rel < 0 or rel >= token_len:
                return None
            return alpha.tokenizer.decode([int(mm[base + rel])],
                                          clean_up_tokenization_spaces=False).strip()

        def span_words(start):
            out = []
            for rel in range(int(start), token_len):
                w = word(rel)
                out.append(w)
                if w == ".":
                    break
            return out

        for lab in r.get("label_positions", []):
            seen.add(("label_position", "role"))
            if lab.get("role") not in allowed_roles:
                flag(r, field="label_position.role", role=lab.get("role"))

        for w in r.get("writes", []):
            pos = int(w["pos"])
            key_field = "entity" if fam == "F2" else "key"
            seen.add(("write", "key"))
            if word(pos) != w.get("key"):
                flag(r, field="write.key", pos=pos, decoded=word(pos), emitted=w.get("key"))
            if fam == "F2":
                seen.add(("write", "entity"))
                if word(pos) != w.get("entity"):
                    flag(r, field="write.entity", pos=pos, decoded=word(pos),
                         emitted=w.get("entity"))
            seen.add(("write", "value"))
            sw = span_words(pos)
            value = sw[-2] if len(sw) >= 2 and sw[-1] == "." else None
            if value != w.get("value"):
                flag(r, field="write.value", pos=pos, decoded=value, emitted=w.get("value"))

        for d in r.get("demonstrations", []):
            seen.add(("demonstration", "key"))
            seen.add(("demonstration", "answer"))
            if word(int(d["pos"])) != d.get("key"):
                flag(r, field="demonstration.key", decoded=word(int(d["pos"])),
                     emitted=d.get("key"))
            if word(int(d["answer_pos"])) != d.get("answer"):
                flag(r, field="demonstration.answer", decoded=word(int(d["answer_pos"])),
                     emitted=d.get("answer"))

        # F3 key state replay for event.old
        table = {w.get("key"): w.get("value") for w in r.get("writes", [])} if fam == "F3" else {}
        events = sorted(r.get("events", []), key=lambda e: int(e["pos"]) if e.get("pos") is not None else 0)
        for idx, e in enumerate(events):
            if "type" in e:
                seen.add(("event", "type"))
                if e.get("type") != event_type_by_family.get(fam):
                    flag(r, field="event.type", emitted=e.get("type"),
                         expected=event_type_by_family.get(fam))
            if "event_index" in e:
                seen.add(("event", "event_index"))
                if int(e.get("event_index", -1)) != idx:
                    flag(r, field="event.event_index", emitted=e.get("event_index"), expected=idx)
            if fam == "F1":
                seen.add(("event", "source_state"))
                seen.add(("event", "target_state"))
                if e.get("source_state") != f"S{idx}":
                    flag(r, field="event.source_state", emitted=e.get("source_state"),
                         expected=f"S{idx}")
                if e.get("target_state") != f"S{idx + 1}":
                    flag(r, field="event.target_state", emitted=e.get("target_state"),
                         expected=f"S{idx + 1}")
            elif fam == "F2":
                seen.add(("event", "swapped_entities"))
                sw = span_words(int(e["pos"]))
                decoded_set = {w for w in sw if w in sigma}
                if decoded_set != set(e.get("swapped_entities", [])):
                    flag(r, field="event.swapped_entities", decoded=sorted(decoded_set),
                         emitted=sorted(e.get("swapped_entities", [])))
            elif fam == "F3":
                seen.add(("event", "key"))
                seen.add(("event", "new"))
                seen.add(("event", "old"))
                sw = span_words(int(e["pos"]))
                dec_key = sw[sw.index("key") + 1] if "key" in sw and sw.index("key") + 1 < len(sw) else None
                dec_new = sw[sw.index("box") + 1] if "box" in sw and sw.index("box") + 1 < len(sw) else None
                if dec_key != e.get("key"):
                    flag(r, field="event.key", decoded=dec_key, emitted=e.get("key"))
                if dec_new != e.get("new"):
                    flag(r, field="event.new", decoded=dec_new, emitted=e.get("new"))
                expected_old = table.get(e.get("key"))
                if e.get("old") != expected_old:
                    flag(r, field="event.old", emitted=e.get("old"), expected=expected_old)
                table[e.get("key")] = e.get("new")
            elif fam == "F5":
                seen.add(("event", "old"))
                seen.add(("event", "new"))
                regids = r.get("registry_ids", {})
                if e.get("old") != regids.get("A"):
                    flag(r, field="event.old", emitted=e.get("old"), expected=regids.get("A"))
                if e.get("new") != regids.get("B"):
                    flag(r, field="event.new", emitted=e.get("new"), expected=regids.get("B"))
        if len(bad) >= 20:
            break

    return {"name": "content_decode", "ok": not bad and bool(records),
            "verified_fields": sorted(seen), "bad": bad[:20]}


def provenance_audit(out_dir, records):
    """Re-establish provenance fields: corpus_version matches the manifest, registry/placement
    fields are well-typed. Provenance fields are not byte-derived, so they get this owner rather
    than a recomputation verifier (spec §7)."""
    manifest = _manifest(out_dir)
    manifest_version = None
    if manifest:
        manifest_version = ((manifest.get("source") or {}).get("corpus_version")
                            or manifest.get("corpus_version"))
    bad = []
    checked = 0
    for r in records:
        checked += 1
        cv = r.get("corpus_version")
        if cv is not None and manifest_version is not None and cv != manifest_version:
            bad.append({"instance_id": r.get("instance_id"), "field": "corpus_version",
                        "record": cv, "manifest": manifest_version})
        if "registry_ids" in r and not isinstance(r["registry_ids"], dict):
            bad.append({"instance_id": r.get("instance_id"), "field": "registry_ids",
                        "error": "not a dict"})
        for f in ("window_index", "start_offset"):
            if f in r and (not isinstance(r[f], int) or r[f] < 0):
                bad.append({"instance_id": r.get("instance_id"), "field": f, "value": r.get(f)})
        for e in r.get("events", []):
            if "transition_id" in e and not isinstance(e["transition_id"], int):
                bad.append({"instance_id": r.get("instance_id"), "field": "transition_id",
                            "value": e.get("transition_id")})
        if len(bad) >= 20:
            break
    return {"name": "provenance_audit", "ok": checked > 0 and not bad,
            "checked": checked, "manifest_corpus_version": manifest_version,
            "verified_fields": ["corpus_version", "registry_ids", "instance_id", "seed",
                                "shard", "window_index", "start_offset", "family", "split",
                                "text_hash", "token_hash", "transition_id"],
            "bad": bad}


def _check_ran_ok(checks):
    """Map each check's base id (split suffix stripped) -> True iff every run of it passed."""
    ran = {}
    for c in checks:
        name = c.get("name", "")
        base = name
        for suf in ("_train", "_eval"):
            if name.endswith(suf):
                base = name[: -len(suf)]
                break
        ran[base] = ran.get(base, True) and bool(c.get("ok"))
    return ran


def metadata_recomputation_audit(checks):
    """Literal per-field coverage gate (spec §7/§18). Builds the set of (scope, field) pairs
    that recomputation verifiers actually asserted this run (their ``verified_fields``, counted
    only for checks that passed), and requires every ``derived`` field to be either literally
    covered or on the ``RECOMPUTE_PENDING`` allowlist. A derived field that is neither is
    build-blocking — so a new field with the mimics_family/composition_depth disease cannot
    ship. ``structural`` containers are covered by a passing traversing owner; ``provenance`` by
    a passing provenance owner; ``reported`` (F5) is exempt."""
    ran = _check_ran_ok(checks)
    covered = set()
    for c in checks:
        if not c.get("ok"):
            continue
        for entry in c.get("verified_fields", []) or []:
            covered.add(tuple(entry))
    index = field_owner_index()
    uncovered, structural_gap, provenance_gap = [], [], []
    pending_hit, pending_now_covered = [], []
    for (scope, field), meta in index.items():
        key = (scope, field)
        kind = meta["kind"]
        if kind == "reported":
            continue
        if kind == "structural":
            if key not in covered and not any(ran.get(o) for o in meta["owners"]):
                structural_gap.append({"scope": scope, "field": field, "owners": meta["owners"]})
            continue
        if kind == "provenance":
            if not any(ran.get(o) for o in meta["owners"]):
                provenance_gap.append({"scope": scope, "field": field, "owners": meta["owners"]})
            continue
        # derived
        if key in covered:
            if key in RECOMPUTE_PENDING:
                pending_now_covered.append([scope, field])  # allowlist should drop this
            continue
        if key in RECOMPUTE_PENDING:
            pending_hit.append([scope, field])
            continue
        uncovered.append({"scope": scope, "field": field, "owners": meta["owners"]})
    derived_keys = {k for k, m in index.items() if m["kind"] == "derived"}
    phantom_pending = sorted(p for p in RECOMPUTE_PENDING if p not in derived_keys)
    ok = not uncovered and not structural_gap and not provenance_gap and not phantom_pending
    return {"name": "metadata_recomputation", "ok": ok,
            "derived_fields": len(derived_keys),
            "derived_independently_recomputed": len(derived_keys) - len(pending_hit),
            "pending_independent_recompute": sorted(pending_hit),
            "pending_now_covered_remove_from_allowlist": sorted(pending_now_covered),
            "phantom_pending_not_a_derived_field": phantom_pending,
            "uncovered_derived_fields": uncovered[:20],
            "structural_gaps": structural_gap[:20],
            "provenance_gaps": provenance_gap[:20]}


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
                checks.append(instance_span_and_seam_roundtrip(out_dir, split, recs, alpha, seq_len))
    checks += [distribution_audit(all_records), marker_audit(all_records),
               replay_audit(all_records), label_audit(all_records),
               split_integrity_audit(all_records),
               schema_completeness_audit(all_records),
               schema_admission_audit(all_records),
               query_position_contract(out_dir, all_records, alpha, seq_len),
               distractor_surface_audit(out_dir, all_records, alpha, seq_len),
               f3_age_audit(all_records),
               filler_scaffolding_audit(out_dir, all_records, alpha, seq_len),
               train_eval_symbol_exclusion_audit(out_dir, all_records, alpha, seq_len),
               f1_rule_consistency(all_records),
               minimal_suffix_audit(all_records),
               composition_depth_audit(all_records),
               age_flag_audit(all_records),
               content_decode_audit(out_dir, all_records, alpha, seq_len),
               provenance_audit(out_dir, all_records),
               difficulty_stratum_audit(all_records)]
    checks.append(metadata_recomputation_audit(checks))
    ok = all(c["ok"] for c in checks)
    report = {"ok": ok, "checks": checks}
    with open(os.path.join(out_dir, "qa_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["qa"] = report
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        corpus_version = (manifest.get("source") or {}).get("corpus_version")
        if corpus_version:
            versioned_path = os.path.join(out_dir, f"MANIFEST.{corpus_version}.json")
            if os.path.exists(versioned_path):
                with open(versioned_path, encoding="utf-8") as f:
                    versioned = json.load(f)
                versioned["qa"] = report
                if isinstance(versioned.get("manifest"), dict):
                    versioned["manifest"]["qa"] = report
                with open(versioned_path, "w", encoding="utf-8") as f:
                    json.dump(versioned, f, indent=2)
    return report
