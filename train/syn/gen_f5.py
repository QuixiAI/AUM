"""F5 eval-only modular stream switch generator."""

from __future__ import annotations

from train.syn.alphabet import F5_WORDS
from train.syn.render import Renderer

NUMBER_WORDS = F5_WORDS[:10]


def _step(x, rule):
    d = rule.data
    return (d["a"] * x + d["b"]) % d["m"]


def _word(n):
    return NUMBER_WORDS[n % 10]


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
    while r.token_len() < target_len - 4:
        x = _step(x, rb)
        r.add(_word(x))
        if rng.random() < 0.12 and r.token_len() < target_len - 4:
            qi = len(queries)
            ans = _word(_step(x, rb))
            queries.append({"pos": None, "answer_pos": None, "answer": ans,
                            "age_from_event": None, "age_bin": age_bin})
            r.add("next", {"role": "query", "path": ("queries", qi, "pos")})
            r.words(":")
            r.add(ans, {"role": "answer", "path": ("queries", qi, "answer_pos")})
    rec = {
        "family": "F5", "split": split, "rule_ids": {"A": rid_a, "B": rid_b},
        "token_len": None, "active_rule_rle": None, "events": events, "writes": [],
        "queries": queries, "distractors": [], "pseudo_event_pos": None,
    }
    ids, rec, text = r.finalize(rec)
    switch_pos = rec["events"][0]["pos"]
    rec["active_rule_rle"] = [[0, rid_a, switch_pos], [switch_pos, rid_b, len(ids) - switch_pos]]
    for q in rec["queries"]:
        q["age_from_event"] = q["answer_pos"] - switch_pos
    rec["token_len"] = len(ids)
    rec["instance_text"] = text
    return ids, rec
