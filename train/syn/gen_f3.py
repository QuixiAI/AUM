"""F3 delayed-correction / long-range recall generator."""

from __future__ import annotations

from train.syn.alphabet import DIGIT_WORDS
from train.syn.render import Renderer


def _age_target(age_bin, rng, max_age):
    raw = int(8 * (3500 / 8) ** ((age_bin + rng.random()) / 10))
    return max(8, min(raw, max(8, max_age)))


def generate(alpha, registry, rng, split, instance_index, target_len, age_bin=0):
    pool = list(alpha.sigma_train if split == "train" else alpha.sigma_eval + alpha.sigma_train)
    rid = rng.choice(list(registry.ids_for_split(split)))
    m = registry.rule_for("F3", rid).data["m"]
    keys = rng.sample(pool, m)
    vals = rng.sample(list(DIGIT_WORDS), m)
    table = dict(zip(keys, vals))

    r = Renderer(alpha)
    r.variant(rng, "binding")
    writes = []
    for key in keys:
        wi = len(writes)
        writes.append({"pos": None, "key": key, "value": table[key]})
        r.words("the", "key")
        r.add(key, {"role": "write", "path": ("writes", wi, "pos")})
        r.words("opens", "box", table[key], ".")

    events = []
    mode = "correction" if rng.random() < 0.5 else "recall"
    if mode == "correction":
        r.bg(rng, rng.randint(4, 24))
        key = rng.choice(keys)
        old = table[key]
        new = rng.choice([v for v in DIGIT_WORDS if v != old])
        table[key] = new
        events.append({"type": "correction", "pos": None, "key": key, "old": old, "new": new})
        r.variant(rng, "correction", {"role": "event", "path": ("events", 0, "pos")})
        r.words("the", "key", key, "now", "opens", "box", new, ".")
        r.bg(rng, _age_target(age_bin, rng, target_len - r.token_len() - 8))
    else:
        r.bg(rng, _age_target(age_bin, rng, target_len - r.token_len() - 8))

    queries = []

    def add_query(key):
        qi = len(queries)
        queries.append({"pos": None, "answer_pos": None, "key": key,
                        "correct_answer": table[key], "answer": table[key],
                        "age_write_to_query": None, "age_correction_to_query": None,
                        "mode": mode, "age_bin": age_bin})
        r.variant(rng, "query", {"role": "query", "path": ("queries", qi, "pos")})
        r.words("key", key, "->")
        r.add(table[key], {"role": "answer", "path": ("queries", qi, "answer_pos")})
        r.words(".")

    add_query(rng.choice(keys))
    for _ in range(rng.randint(2, 5)):
        r.bg(rng, rng.randint(0, 4))
        add_query(rng.choice(keys))

    rec = {
        "family": "F3", "split": split, "rule_ids": {"A": rid}, "token_len": None,
        "active_rule_rle": None, "events": events, "writes": writes, "queries": queries,
        "distractors": [], "pseudo_event_pos": None,
    }
    ids, rec, text = r.finalize(rec)
    rec["active_rule_rle"] = [[0, rid, len(ids)]]
    write_pos_by_key = {w["key"]: w["pos"] for w in rec["writes"]}
    for q in rec["queries"]:
        q["age_write_to_query"] = q["answer_pos"] - write_pos_by_key[q["key"]]
        q["age_correction_to_query"] = (
            None if not rec["events"] else q["answer_pos"] - rec["events"][-1]["pos"])
    rec["token_len"] = len(ids)
    rec["instance_text"] = text
    return ids, rec
