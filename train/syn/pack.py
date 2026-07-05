"""Window-aware SYN packing and manifest writing."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np

from train.syn import CORPUS_VERSION
from train.syn.alphabet import stable_seed
from train.syn import gen_f1, gen_f2, gen_f3, gen_f4, gen_f5
from train.syn.render import RenderRejected

DTYPE = np.uint16
GENS = {"F1": gen_f1.generate, "F2": gen_f2.generate, "F3": gen_f3.generate,
        "F4": gen_f4.generate, "F5": gen_f5.generate}
TRAIN_SHARES = {"F1": 0.30, "F2": 0.25, "F3": 0.28, "F4": 0.17}
MAX_INSTANCES_PER_WINDOW = 32
WINDOW_TASK_FRACTION_TARGET = (0.45, 0.60)
WINDOW_TASK_FRACTION_QA = (0.35, 0.70)


@dataclass
class ShardWriter:
    out_dir: str
    split: str
    shard_size_tokens: int
    buf: list[int] = field(default_factory=list)
    shards: list[dict] = field(default_factory=list)
    total: int = 0

    @property
    def shard_name(self) -> str:
        return f"{self.split}_{len(self.shards):05d}.bin"

    def add_window(self, ids: list[int]) -> tuple[str, int]:
        if len(self.buf) + len(ids) > self.shard_size_tokens and self.buf:
            self.flush()
        shard = self.shard_name
        window_index = len(self.buf) // 4096
        self.buf.extend(ids)
        self.total += len(ids)
        return shard, window_index

    def flush(self):
        if not self.buf:
            return
        arr = np.asarray(self.buf, dtype=DTYPE)
        name = self.shard_name
        arr.tofile(os.path.join(self.out_dir, name))
        self.shards.append({"name": name, "n_tokens": int(arr.size)})
        self.buf = []

    def close(self):
        self.flush()


class SidecarWriter:
    def __init__(self, out_dir, split):
        os.makedirs(os.path.join(out_dir, "sidecars"), exist_ok=True)
        self.path = os.path.join(out_dir, "sidecars", f"{split}.jsonl.gz")
        self.f = gzip.open(self.path, "wt", encoding="utf-8")
        self.count = 0

    def write(self, rec):
        self.f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
        self.count += 1

    def close(self):
        self.f.close()


def log_uniform_len(rng, lo=256, hi=3968):
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


def target_len_for_family(rng, family, age_bin):
    if family == "F3":
        if age_bin >= 7:
            return rng.randint(1500, 1800)
        if age_bin >= 5:
            return rng.randint(900, 1200)
        return rng.randint(320, 640)
    if family == "F5":
        return rng.randint(160, 320)
    return rng.randint(256, 640)


def bg_ids(alpha, rng, n):
    return [alpha.single_id(w) for w in alpha.bg_words(rng, n)]


def make_instance(alpha, registry, family, split, index, seed, target_len=None, age_bin=None):
    rng = random.Random(stable_seed(CORPUS_VERSION, family, split, index, seed))
    target_len = target_len or log_uniform_len(rng)
    age_bin = index % 10 if age_bin is None else age_bin
    ids, rec = GENS[family](alpha, registry, rng, split, index, target_len, age_bin)
    if any(t < 0 or t > np.iinfo(DTYPE).max for t in ids):
        raise ValueError(f"{family} emitted token outside uint16 range")
    text = rec.pop("instance_text")
    ids2 = alpha.tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids2 != ids:
        raise RenderRejected(f"{family}: tokenize(instance_text) != emitted ids")
    rec.update({
        "instance_id": f"{family.lower()}-s{index:09d}",
        "corpus_version": CORPUS_VERSION,
        "seed": int(stable_seed(CORPUS_VERSION, family, split, index, seed)),
        "token_len": len(ids),
        "token_hash": hashlib.sha256(np.asarray(ids, dtype=DTYPE).tobytes()).hexdigest(),
    })
    return ids, rec


def _next_family(targets, produced, allow_over_target=False):
    families = [f for f, target in targets.items()
                if allow_over_target or produced.get(f, 0) < target]
    if not families:
        return None
    return min(families, key=lambda f: produced.get(f, 0) / max(1, targets[f]))


def _fit_window(alpha, registry, split, targets, produced, counters, seed, seq_len=4096):
    rng = random.Random(stable_seed("window", split, sum(counters.values()), seed))
    window = []
    records = []
    rejects = 0
    local_produced = dict(produced)
    task_tokens = 0
    target_task_fraction = rng.uniform(*WINDOW_TASK_FRACTION_TARGET)
    while len(records) < MAX_INSTANCES_PER_WINDOW:
        if records and task_tokens / seq_len >= target_task_fraction:
            break
        remaining = seq_len - len(window) - 1
        if remaining < 96:
            break
        family = _next_family(targets, local_produced, allow_over_target=bool(records))
        if family is None:
            break
        idx = counters[family]
        age_bin = idx % 10
        target = min(target_len_for_family(rng, family, age_bin), remaining)
        for attempt in range(25):
            try:
                ids, rec = make_instance(alpha, registry, family, split, idx + attempt, seed,
                                         target_len=target, age_bin=age_bin)
                if len(ids) > remaining:
                    continue
                counters[family] = idx + attempt + 1
                break
            except RenderRejected:
                rejects += 1
        else:
            if records:
                break
            raise RuntimeError(f"too many rejected SYN instances for {family}/{split}")
        start = len(window)
        window.extend(ids)
        records.append((start, ids, rec))
        task_tokens += int(rec.get("task_token_count", len(ids)))
        local_produced[family] = local_produced.get(family, 0) + len(ids)
        if len(window) < seq_len:
            window.append(alpha.eos_id)
    if len(window) < seq_len:
        window.extend([alpha.eos_id] * (seq_len - len(window)))
    return window[:seq_len], records, rejects


def _family_queue_for_targets(targets, produced):
    families = [f for f, target in targets.items() if produced.get(f, 0) < target]
    if not families:
        return []
    return sorted(families, key=lambda f: produced.get(f, 0) / max(1, targets[f]))


def generate_split(alpha, registry, out_dir, split, targets, seed=1337,
                   shard_size_tokens=100_000_000, seq_len=4096):
    writer = ShardWriter(out_dir, split, shard_size_tokens)
    sidecar = SidecarWriter(out_dir, split)
    counters = {f: 0 for f in targets}
    produced = {f: 0 for f in targets}
    task_produced = {f: 0 for f in targets}
    windows = 0
    rejected = 0
    try:
        while True:
            queue = _family_queue_for_targets(targets, produced)
            if not queue:
                break
            window, records, rejects = _fit_window(alpha, registry, split, targets, produced,
                                                  counters, seed, seq_len)
            rejected += rejects
            shard, window_index = writer.add_window(window)
            windows += 1
            for start, ids, rec in records:
                rec["shard"] = shard
                rec["window_index"] = window_index
                rec["start_offset"] = start
                sidecar.write(rec)
                produced[rec["family"]] += len(ids)
                task_produced[rec["family"]] += int(rec.get("task_token_count", len(ids)))
    finally:
        writer.close()
        sidecar.close()
    total_instances = sidecar.count + rejected
    reject_rate = rejected / total_instances if total_instances else 0.0
    if reject_rate > 0.01:
        raise RuntimeError(f"SYN roundtrip rejection rate too high for {split}: {reject_rate:.2%}")
    return {"writer": writer, "sidecar": sidecar.path, "sidecar_records": sidecar.count,
            "family_tokens": produced, "family_task_tokens": task_produced,
            "windows": windows, "roundtrip_rejected": rejected,
            "roundtrip_rejection_rate": reject_rate}


def write_manifest(out_dir, tokenizer, vocab_size, eos_id, seq_len, split_results, source,
                   tokenizer_identity=None):
    splits = {}
    for split, result in split_results.items():
        w = result["writer"]
        splits[split] = {"shards": w.shards, "total_tokens": w.total,
                         "approx_sequences": w.total // seq_len}
    manifest = {
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "eos_id": int(eos_id),
        "seq_len": seq_len,
        "source": source,
        "packing_policy": {
            "unit": "fixed_window",
            "window_tokens": seq_len,
            "max_synthetic_instances_per_window": MAX_INSTANCES_PER_WINDOW,
            "window_background": "eos_padding",
            "window_task_fraction_target": list(WINDOW_TASK_FRACTION_TARGET),
            "window_task_fraction_qa": list(WINDOW_TASK_FRACTION_QA),
            "instance_boundary_mask": False,
            "eos_between_instances": True,
            "note": "SYN shards are standalone synthetic windows; sidecar records identify task-bearing spans.",
        },
        "tokenizer_identity": tokenizer_identity,
        "splits": splits,
        "created_unix": int(time.time()),
        "read_hint": "np.memmap(<out_dir>/<shard>, dtype=np.uint16, mode='r') -> 1-D token stream",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def run_generation(alpha, registry, out_dir, tokenizer_name, vocab_size, seq_len=4096,
                   shard_size_tokens=100_000_000, train_tokens=1_000_000_000,
                   eval_tokens=40_000_000, seed=1337):
    os.makedirs(out_dir, exist_ok=True)
    train_targets = {f: int(train_tokens * share) for f, share in TRAIN_SHARES.items()}
    eval_fams = ("F1", "F2", "F3", "F4", "F5")
    eval_targets = {f: max(4096, eval_tokens // len(eval_fams)) for f in eval_fams}
    results = {
        "train": generate_split(alpha, registry, out_dir, "train", train_targets, seed,
                                shard_size_tokens, seq_len),
        "eval": generate_split(alpha, registry, out_dir, "eval", eval_targets, seed,
                               shard_size_tokens, seq_len),
    }
    manifest = write_manifest(out_dir, tokenizer_name, vocab_size, alpha.eos_id, seq_len, results,
                              {"corpus_version": CORPUS_VERSION, "seed": seed,
                               "train_targets": train_targets, "eval_targets": eval_targets},
                              tokenizer_identity=None if alpha.identity is None else alpha.identity.__dict__)
    summary = {"corpus_version": CORPUS_VERSION, "registry_hash": registry.hash(),
               "splits": {k: {"family_tokens": v["family_tokens"],
                              "family_task_tokens": v["family_task_tokens"],
                              "total_tokens": v["writer"].total,
                              "sidecar_records": v["sidecar_records"],
                              "windows": v["windows"],
                              "roundtrip_rejected": v["roundtrip_rejected"],
                              "roundtrip_rejection_rate": v["roundtrip_rejection_rate"]}
                          for k, v in results.items()},
               "tokenizer_identity": None if alpha.identity is None else alpha.identity.__dict__}
    with open(os.path.join(out_dir, f"MANIFEST.{CORPUS_VERSION}.json"), "w", encoding="utf-8") as f:
        json.dump({**summary, "manifest": manifest}, f, indent=2)
    return summary
