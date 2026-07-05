"""F4 flat-null generator."""

from __future__ import annotations

from train.syn import MAX_ATTENTION_WINDOW_PLANNED
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
    r.bg(rng, rng.randint(3, 12))
    distractors = [{"pos": None, "mimics_family": shape, "variant": f"{shape}_surface"}]
    r.variant(rng, f"distractor_{shape}",
              {"role": "distractor", "path": ("distractors", 0, "pos")})

    queries = []
    source_by_key = {key: ("write", key) for key in base}
    query_meta = []
    for _ in range(rng.randint(8, 12)):
        r.bg(rng, rng.randint(0, 4))
        key = rng.choice(base)
        qi = len(queries)
        source_ref = source_by_key[key]
        queries.append({
            "pos": None,
            "answer_pos": None,
            "key": key,
            "answer": mapping[key],
            "prediction_anchor": "->",
            "target_age_bin": age_bin,
            "composition_depth": 0,
            "evidence_positions": None,
            "minimal_sufficient_suffix_len": None,
            "hard_suffix": None,
        })
        r.variant(rng, "query", {"role": "query", "path": ("queries", qi, "pos")})
        r.words(key, "->")
        r.add(mapping[key], {"role": "answer", "path": ("queries", qi, "answer_pos")})
        r.words(".")
        query_meta.append({"source_ref": source_ref})
        source_by_key[key] = ("query", qi)

    rec = {
        "family": "F4",
        "split": split,
        "registry_ids": {"base_rule": rid},
        "composition_depth": 0,
        "token_len": None,
        "active_map_rle": None,
        "events": [],
        "writes": writes,
        "queries": queries,
        "distractors": distractors,
        "pseudo_event_pos": None,
    }
    ids, rec, text = r.finalize(rec)
    rec["pseudo_event_pos"] = rec["distractors"][0]["pos"]
    rec["active_map_rle"] = [{"start": 0, "end": len(ids), "map": dict(mapping)}]
    write_pos = {w["key"]: w["pos"] for w in rec["writes"]}
    for q, meta in zip(rec["queries"], query_meta):
        kind, key_or_index = meta["source_ref"]
        source_pos = write_pos[key_or_index] if kind == "write" else rec["queries"][key_or_index]["pos"]
        evidence = [int(source_pos)]
        pred_pos = q["answer_pos"] - 1
        q["evidence_positions"] = evidence
        q["minimal_sufficient_suffix_len"] = int(pred_pos - evidence[0] + 1)
        q["hard_suffix"] = q["minimal_sufficient_suffix_len"] > MAX_ATTENTION_WINDOW_PLANNED
    rec["token_len"] = len(ids)
    rec["instance_text"] = text
    return ids, rec
