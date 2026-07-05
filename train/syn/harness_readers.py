"""Readers for SYN sidecars and metric-position conventions."""

from __future__ import annotations

import gzip
import json


def open_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def prediction_pos_for_answer(query: dict) -> int:
    return int(query["answer_pos"]) - 1


def iter_query_positions(records):
    for rec in records:
        for q in rec.get("queries", []):
            yield rec, q, prediction_pos_for_answer(q)


def expand_active_rule_rle(record: dict):
    out = []
    for start, rule_id, length in record.get("active_rule_rle", []):
        out.extend([rule_id] * length)
    return out

