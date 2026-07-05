"""F2 binding-swap generator."""

from __future__ import annotations

from train.syn import MAX_ATTENTION_WINDOW_PLANNED
from train.syn.render import Renderer


def _active_map_rle(state_maps, event_positions, total_len):
    out = []
    for i, mapping in enumerate(state_maps):
        start = 0 if i == 0 else event_positions[i - 1]
        end = total_len if i == len(state_maps) - 1 else event_positions[i]
        out.append({"start": int(start), "end": int(end), "map": dict(mapping)})
    return out


def _fill_query_evidence(rec, query_meta):
    event_positions = [e["pos"] for e in rec["events"]]
    write_pos = {w["key"]: w["pos"] for w in rec["writes"]}

    def ref_pos(ref):
        kind, key_or_index = ref
        if kind == "write":
            return write_pos[key_or_index]
        if kind == "query":
            return rec["queries"][key_or_index]["pos"]
        return event_positions[key_or_index]

    for q, meta in zip(rec["queries"], query_meta):
        events_seen = meta["events_seen"]
        source_pos = ref_pos(meta["source_ref"])
        evidence = [source_pos]
        for event_index in range(events_seen):
            event_pos = event_positions[event_index]
            if event_pos > source_pos:
                evidence.append(event_pos)
        evidence = sorted(set(int(p) for p in evidence))
        pred_pos = q["answer_pos"] - 1
        q["evidence_positions"] = evidence
        q["minimal_sufficient_suffix_len"] = int(pred_pos - min(evidence) + 1)
        q["hard_suffix"] = q["minimal_sufficient_suffix_len"] > MAX_ATTENTION_WINDOW_PLANNED
        q["age_from_binding_statement"] = int(q["answer_pos"] - write_pos[q["entity"]])
        q["age_from_swap"] = (
            None if events_seen == 0 else int(q["answer_pos"] - event_positions[events_seen - 1])
        )
        latest = set(rec["events"][events_seen - 1]["swapped_entities"]) if events_seen else set()
        q["entity_was_swapped"] = q["entity"] in latest
        q["entity_ever_swapped"] = any(
            q["entity"] in set(rec["events"][i]["swapped_entities"])
            for i in range(events_seen)
        )


def generate(alpha, registry, rng, split, instance_index, target_len, age_bin=0):
    pool = list(alpha.sigma_train if split == "train" else alpha.sigma_eval + alpha.sigma_train)
    rid = rng.choice(list(registry.ids_for_split(split)))
    k = registry.rule_for("F2", rid).data["k"]
    entities = rng.sample(pool, k)
    attrs = rng.sample([x for x in pool if x not in entities], k)
    values = dict(zip(entities, attrs))
    state_maps = [dict(values)]
    source_by_entity = {ent: ("write", ent) for ent in entities}
    composition_depth = 2 if instance_index % 3 == 0 and k >= 3 else 1

    r = Renderer(alpha)
    r.variant(rng, "binding")
    writes = []
    for ent in entities:
        wi = len(writes)
        writes.append({"pos": None, "key": ent, "entity": ent, "value": values[ent]})
        r.add(ent, {"role": "write", "path": ("writes", wi, "pos")})
        r.words("is", values[ent], ".")

    queries = []
    query_meta = []

    def add_query(ent, events_seen):
        qi = len(queries)
        ans = values[ent]
        source_ref = source_by_entity[ent]
        queries.append({
            "pos": None,
            "answer_pos": None,
            "entity": ent,
            "answer": ans,
            "prediction_anchor": "->",
            "entity_was_swapped": None,
            "entity_ever_swapped": None,
            "age_from_swap": None,
            "age_from_binding_statement": None,
            "target_age_bin": age_bin,
            "composition_depth": composition_depth,
            "evidence_positions": None,
            "minimal_sufficient_suffix_len": None,
            "hard_suffix": None,
        })
        query_meta.append({"source_ref": source_ref, "events_seen": events_seen})
        r.variant(rng, "query", {"role": "query", "path": ("queries", qi, "pos")})
        r.words(ent, "->")
        r.add(ans, {"role": "answer", "path": ("queries", qi, "answer_pos")})
        r.words(".")
        source_by_entity[ent] = ("query", qi)

    for _ in range(2):
        r.bg(rng, rng.randint(0, 3))
        add_query(rng.choice(entities), events_seen=0)

    events = []
    all_swapped = set()
    for event_index in range(composition_depth):
        r.bg(rng, rng.randint(3, 12))
        events.append({"type": "swap", "pos": None, "event_index": event_index,
                       "swapped_entities": None})
        r.variant(rng, "correction", {"role": "event", "path": ("events", event_index, "pos")})
        if rng.random() < 0.25 and k >= 3:
            swapped = rng.sample(entities, 3)
            values[swapped[0]], values[swapped[1]], values[swapped[2]] = (
                values[swapped[2]], values[swapped[0]], values[swapped[1]]
            )
            r.words(swapped[0], swapped[1], "and", swapped[2], "were", "exchanged", ".")
        else:
            swapped = rng.sample(entities, 2)
            values[swapped[0]], values[swapped[1]] = values[swapped[1]], values[swapped[0]]
            r.words(swapped[0], "and", swapped[1], "were", "exchanged", ".")
        events[event_index]["swapped_entities"] = list(swapped)
        all_swapped.update(swapped)
        state_maps.append(dict(values))

        for _ in range(rng.randint(1, 3)):
            r.bg(rng, rng.randint(0, 3))
            add_query(rng.choice(entities), events_seen=event_index + 1)

    unswapped = [e for e in entities if e not in all_swapped] or entities
    post_keys = [rng.choice(unswapped)]
    total_queries = rng.randint(7, 12)
    while len(queries) + len(post_keys) < total_queries:
        post_keys.append(rng.choice(entities))
    for ent in post_keys:
        r.bg(rng, rng.randint(0, 4))
        add_query(ent, events_seen=composition_depth)

    rec = {
        "family": "F2",
        "split": split,
        "registry_ids": {"base_rule": rid},
        "composition_depth": composition_depth,
        "token_len": None,
        "active_map_rle": None,
        "events": events,
        "writes": writes,
        "queries": queries,
        "distractors": [],
        "pseudo_event_pos": None,
    }
    ids, rec, text = r.finalize(rec)
    event_positions = [e["pos"] for e in rec["events"]]
    rec["active_map_rle"] = _active_map_rle(state_maps, event_positions, len(ids))
    _fill_query_evidence(rec, query_meta)
    rec["token_len"] = len(ids)
    rec["instance_text"] = text
    return ids, rec
