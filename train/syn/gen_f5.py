"""F5 eval-only modular stream switch generator."""

from __future__ import annotations

from train.syn import MAX_ATTENTION_WINDOW_PLANNED
from train.syn.alphabet import F5_WORDS
from train.syn.render import Renderer

NUMBER_WORDS = F5_WORDS[:10]


def _step(x, rule):
    d = rule.data
    return (d["a"] * x + d["b"]) % d["m"]


def _word(n):
    return NUMBER_WORDS[n % 10]


def _rule_payload(rule):
    return {"m": rule.data["m"], "a": rule.data["a"], "b": rule.data["b"]}


def generate(alpha, registry, rng, split, instance_index, target_len, age_bin=0):
    ids_for_split = list(registry.ids_for_split(split))
    rid_a = rng.choice(ids_for_split)
    rid_b = rng.choice([x for x in ids_for_split if x != rid_a])
    ra, rb = registry.rule_for("F5", rid_a), registry.rule_for("F5", rid_b)
    r = Renderer(alpha)
    x = rng.randrange(0, min(ra.data["m"], 10))
    switch_target = int(target_len * rng.uniform(0.35, 0.65))
    while r.token_len() < switch_target:
        r.add(_word(x))
        x = _step(x, ra)
    events = [{"type": "switch", "pos": None, "old": rid_a, "new": rid_b}]
    r.add("switch", {"role": "event", "path": ("events", 0, "pos")})
    x = rng.randrange(0, min(rb.data["m"], 10))
    queries = []

    def add_query():
        qi = len(queries)
        ans = _word(_step(x, rb))
        queries.append({
            "pos": None,
            "answer_pos": None,
            "answer": ans,
            "prediction_anchor": ":",
            "age_from_event": None,
            "target_age_bin": age_bin,
            "composition_depth": 1,
            "evidence_positions": None,
            "minimal_sufficient_suffix_len": None,
            "hard_suffix": None,
        })
        r.add("next", {"role": "query", "path": ("queries", qi, "pos")})
        r.words(":")
        r.add(ans, {"role": "answer", "path": ("queries", qi, "answer_pos")})

    while r.token_len() < target_len - 4:
        x = _step(x, rb)
        r.add(_word(x))
        if rng.random() < 0.12 and r.token_len() < target_len - 4:
            add_query()
    if not queries and r.token_len() < target_len - 2:
        add_query()

    rec = {
        "family": "F5",
        "split": split,
        "registry_ids": {"A": rid_a, "B": rid_b},
        "composition_depth": 1,
        "token_len": None,
        "active_param_rle": None,
        "events": events,
        "writes": [],
        "queries": queries,
        "distractors": [],
        "pseudo_event_pos": None,
    }
    ids, rec, text = r.finalize(rec)
    switch_pos = rec["events"][0]["pos"]
    rec["active_param_rle"] = [
        {"start": 0, "end": switch_pos, "params": _rule_payload(ra)},
        {"start": switch_pos, "end": len(ids), "params": _rule_payload(rb)},
    ]
    for q in rec["queries"]:
        pred_pos = q["answer_pos"] - 1
        q["age_from_event"] = int(q["answer_pos"] - switch_pos)
        q["evidence_positions"] = [int(switch_pos)]
        q["minimal_sufficient_suffix_len"] = int(pred_pos - switch_pos + 1)
        q["hard_suffix"] = q["minimal_sufficient_suffix_len"] > MAX_ATTENTION_WINDOW_PLANNED
    rec["token_len"] = len(ids)
    rec["instance_text"] = text
    return ids, rec
