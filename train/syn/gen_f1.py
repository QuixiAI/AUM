"""F1 branch-reversal generator."""

from __future__ import annotations

from train.syn.render import Renderer


def _mapping(rule, working):
    return {working[i]: working[rule.data["perm"][i]] for i in range(8)}


def _pair(r, key, value):
    r.words(key, "->", value, ".")


def generate(alpha, registry, rng, split, instance_index, target_len, age_bin=0):
    pool = list(alpha.sigma_train if split == "train" else alpha.sigma_eval + alpha.sigma_train)
    working = rng.sample(pool, 8)
    ids_for_split = list(registry.ids_for_split(split))
    rid_a = rng.choice(ids_for_split)
    rid_b = rng.choice([x for x in ids_for_split if x != rid_a])
    map_a = _mapping(registry.rule_for("F1", rid_a), working)
    map_b = _mapping(registry.rule_for("F1", rid_b), working)

    r = Renderer(alpha)
    r.variant(rng, "rule")
    writes = []
    queries = []
    for key, value in map_a.items():
        wi = len(writes)
        writes.append({"pos": None, "key": key, "value": value})
        r.add(key, {"role": "write", "path": ("writes", wi, "pos")})
        r.words("means", value, ".")

    for demo_i in range(rng.randint(8, 14)):
        r.bg(rng, rng.randint(0, 3))
        key = rng.choice(working)
        if demo_i < 2:
            qi = len(queries)
            queries.append({"pos": None, "answer_pos": None, "key": key, "answer": map_a[key],
                            "age_from_event": None, "age_from_write": None, "age_bin": age_bin})
            r.variant(rng, "query", {"role": "query", "path": ("queries", qi, "pos")})
            r.words(key, "->")
            r.add(map_a[key], {"role": "answer", "path": ("queries", qi, "answer_pos")})
            r.words(".")
        else:
            _pair(r, key, map_a[key])

    r.bg(rng, rng.randint(4, 24))
    restatement = "partial" if rng.random() < 0.5 else "full"
    events = [{"type": "reversal", "pos": None, "old": rid_a, "new": rid_b,
               "restatement": restatement}]
    r.variant(rng, "correction", {"role": "event", "path": ("events", 0, "pos")})
    for key in (rng.sample(working, 3) if restatement == "partial" else working):
        r.words(key, "now", "means", map_b[key], ".")

    for _ in range(rng.randint(8, 14)):
        r.bg(rng, rng.randint(0, 4))
        key = rng.choice(working)
        _pair(r, key, map_b[key])

    max_queries = rng.randint(6, 10)
    changed = [k for k in working if map_b[k] != map_a[k]] or working
    while len(queries) < max_queries:
        r.bg(rng, rng.randint(0, 4))
        key = rng.choice(changed)
        qi = len(queries)
        queries.append({"pos": None, "answer_pos": None, "key": key, "answer": map_b[key],
                        "age_from_event": None, "age_from_write": None, "age_bin": age_bin})
        r.variant(rng, "query", {"role": "query", "path": ("queries", qi, "pos")})
        r.words(key, "->")
        r.add(map_b[key], {"role": "answer", "path": ("queries", qi, "answer_pos")})
        r.words(".")

    rec = {
        "family": "F1", "split": split, "rule_ids": {"A": rid_a, "B": rid_b},
        "token_len": None, "active_rule_rle": None, "events": events, "writes": writes,
        "queries": queries, "distractors": [], "pseudo_event_pos": None,
    }
    ids, rec, text = r.finalize(rec)
    event_pos = rec["events"][0]["pos"]
    rec["active_rule_rle"] = [[0, rid_a, event_pos], [event_pos, rid_b, len(ids) - event_pos]]
    for q in rec["queries"]:
        q["age_from_event"] = q["answer_pos"] - event_pos
        q["age_from_write"] = q["answer_pos"]
    rec["token_len"] = len(ids)
    rec["instance_text"] = text
    return ids, rec
