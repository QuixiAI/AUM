"""F2 binding-swap generator."""

from __future__ import annotations

from train.syn.render import Renderer


def generate(alpha, registry, rng, split, instance_index, target_len, age_bin=0):
    pool = list(alpha.sigma_train if split == "train" else alpha.sigma_eval + alpha.sigma_train)
    rid = rng.choice(list(registry.ids_for_split(split)))
    k = registry.rule_for("F2", rid).data["k"]
    entities = rng.sample(pool, k)
    attrs = rng.sample([x for x in pool if x not in entities], k)
    values = dict(zip(entities, attrs))

    r = Renderer(alpha)
    r.variant(rng, "binding")
    writes = []
    for ent in entities:
        wi = len(writes)
        writes.append({"pos": None, "key": ent, "value": values[ent]})
        r.add(ent, {"role": "write", "path": ("writes", wi, "pos")})
        r.words("is", values[ent], ".")

    queries = []
    for _ in range(rng.randint(3, 6)):
        r.bg(rng, rng.randint(0, 4))
        ent = rng.choice(entities)
        qi = len(queries)
        queries.append({"pos": None, "answer_pos": None, "entity": ent,
                        "correct_answer": values[ent], "answer": values[ent],
                        "entity_was_swapped": None, "age_from_swap": None,
                        "age_from_binding_statement": None, "age_bin": age_bin})
        r.variant(rng, "query", {"role": "query", "path": ("queries", qi, "pos")})
        r.words(ent, "->")
        r.add(values[ent], {"role": "answer", "path": ("queries", qi, "answer_pos")})
        r.words(".")

    r.bg(rng, rng.randint(4, 24))
    events = [{"type": "swap", "pos": None, "swapped_entities": None}]
    r.variant(rng, "correction", {"role": "event", "path": ("events", 0, "pos")})
    if rng.random() < 0.25 and k >= 3:
        swapped = rng.sample(entities, 3)
        values[swapped[0]], values[swapped[1]], values[swapped[2]] = (
            values[swapped[2]], values[swapped[0]], values[swapped[1]])
        r.words(swapped[0], swapped[1], "and", swapped[2], "were", "exchanged", ".")
    else:
        swapped = rng.sample(entities, 2)
        values[swapped[0]], values[swapped[1]] = values[swapped[1]], values[swapped[0]]
        r.words(swapped[0], "and", swapped[1], "were", "exchanged", ".")
    events[0]["swapped_entities"] = swapped

    for _ in range(rng.randint(7, 12)):
        r.bg(rng, rng.randint(0, 5))
        ent = rng.choice(entities)
        qi = len(queries)
        queries.append({"pos": None, "answer_pos": None, "entity": ent,
                        "correct_answer": values[ent], "answer": values[ent],
                        "entity_was_swapped": ent in swapped, "age_from_swap": None,
                        "age_from_binding_statement": None, "age_bin": age_bin})
        r.variant(rng, "query", {"role": "query", "path": ("queries", qi, "pos")})
        r.words(ent, "->")
        r.add(values[ent], {"role": "answer", "path": ("queries", qi, "answer_pos")})
        r.words(".")

    rec = {
        "family": "F2", "split": split, "rule_ids": {"A": rid}, "token_len": None,
        "active_rule_rle": None, "events": events, "writes": writes, "queries": queries,
        "distractors": [], "pseudo_event_pos": None,
    }
    ids, rec, text = r.finalize(rec)
    event_pos = rec["events"][0]["pos"]
    rec["active_rule_rle"] = [[0, rid, len(ids)]]
    for q in rec["queries"]:
        q["entity_was_swapped"] = q["entity"] in swapped
        q["age_from_swap"] = q["answer_pos"] - event_pos
        q["age_from_binding_statement"] = q["answer_pos"]
    rec["token_len"] = len(ids)
    rec["instance_text"] = text
    return ids, rec
