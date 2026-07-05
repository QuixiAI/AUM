"""F4 flat-null generator."""

from __future__ import annotations

from train.syn.render import Renderer


def generate(alpha, registry, rng, split, instance_index, target_len, age_bin=0):
    pool = list(alpha.sigma_train if split == "train" else alpha.sigma_eval + alpha.sigma_train)
    rid = rng.choice(list(registry.ids_for_split(split)))
    shape = registry.rule_for("F4", rid).kind
    base = rng.sample(pool, 8)
    mapping = {base[i]: base[(i + 1) % len(base)] for i in range(len(base))}

    r = Renderer(alpha)
    r.variant(rng, "rule")
    writes = []
    for key, value in mapping.items():
        wi = len(writes)
        writes.append({"pos": None, "key": key, "value": value})
        r.add(key, {"role": "write", "path": ("writes", wi, "pos")})
        r.words("means", value, ".")
    r.bg(rng, rng.randint(4, 24))
    distractors = [{"pos": None, "mimics_family": shape, "variant": "same"}]
    r.variant(rng, "distractor", {"role": "distractor", "path": ("distractors", 0, "pos")})

    queries = []
    for _ in range(rng.randint(8, 12)):
        r.bg(rng, rng.randint(0, 5))
        key = rng.choice(base)
        qi = len(queries)
        queries.append({"pos": None, "answer_pos": None, "key": key,
                        "answer": mapping[key], "age_bin": age_bin})
        r.variant(rng, "query", {"role": "query", "path": ("queries", qi, "pos")})
        r.words(key, "->")
        r.add(mapping[key], {"role": "answer", "path": ("queries", qi, "answer_pos")})
        r.words(".")

    rec = {
        "family": "F4", "split": split, "rule_ids": {"A": rid}, "token_len": None,
        "active_rule_rle": None, "events": [], "writes": writes, "queries": queries,
        "distractors": distractors, "pseudo_event_pos": None,
    }
    ids, rec, text = r.finalize(rec)
    rec["pseudo_event_pos"] = rec["distractors"][0]["pos"]
    rec["active_rule_rle"] = [[0, rid, len(ids)]]
    rec["token_len"] = len(ids)
    rec["instance_text"] = text
    return ids, rec
