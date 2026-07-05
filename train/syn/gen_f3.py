"""F3 delayed-correction / long-range recall generator."""

from __future__ import annotations

from train.syn import MAX_ATTENTION_WINDOW_PLANNED
from train.syn.alphabet import DIGIT_WORDS
from train.syn.render import Renderer


def _age_target(age_bin, rng, max_age):
    raw = int(8 * (3500 / 8) ** ((age_bin + rng.random()) / 10))
    return max(8, min(raw, max(8, max_age)))


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
        q["age_write_to_query"] = int(q["answer_pos"] - write_pos[q["key"]])
        q["age_correction_to_query"] = (
            None if not event_positions else int(q["answer_pos"] - event_positions[-1])
        )


def generate(alpha, registry, rng, split, instance_index, target_len, age_bin=0):
    pool = list(alpha.sigma_train if split == "train" else alpha.sigma_eval + alpha.sigma_train)
    rid = rng.choice(list(registry.ids_for_split(split)))
    m = registry.rule_for("F3", rid).data["m"]
    keys = rng.sample(pool, m)
    vals = rng.sample(list(DIGIT_WORDS), m)
    table = dict(zip(keys, vals))
    source_by_key = {key: ("write", key) for key in keys}
    state_maps = [dict(table)]

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
    controlled_gaps = []
    mode = "correction" if instance_index % 2 else "recall"
    composition_depth = 1
    if mode == "correction":
        composition_depth = 2 if instance_index % 4 == 1 else 1
        r.bg(rng, rng.randint(3, 12))
        for event_index in range(composition_depth):
            key = rng.choice(keys)
            old = table[key]
            new = rng.choice([v for v in DIGIT_WORDS if v != old])
            table[key] = new
            events.append({
                "type": "correction",
                "pos": None,
                "key": key,
                "old": old,
                "new": new,
                "event_index": event_index,
            })
            r.variant(rng, "correction", {"role": "event", "path": ("events", event_index, "pos")})
            r.words("the", "key", key, "now", "opens", "box", new, ".")
            source_by_key[key] = ("event", event_index)
            state_maps.append(dict(table))
            if event_index + 1 < composition_depth:
                r.bg(rng, rng.randint(2, 8))

    max_gap = target_len - r.token_len() - 48
    gap = _age_target(age_bin, rng, max_gap)
    controlled_gaps.append({"kind": "age", "target": gap, "realized": gap,
                            "target_age_bin": age_bin})
    r.bg(rng, gap, controlled=True)

    queries = []
    query_meta = []

    def add_query(key):
        qi = len(queries)
        ans = table[key]
        source_ref = source_by_key[key]
        queries.append({
            "pos": None,
            "answer_pos": None,
            "key": key,
            "answer": ans,
            "prediction_anchor": "->",
            "age_write_to_query": None,
            "age_correction_to_query": None,
            "mode": mode,
            "target_age_bin": age_bin,
            "composition_depth": composition_depth,
            "evidence_positions": None,
            "minimal_sufficient_suffix_len": None,
            "hard_suffix": None,
        })
        query_meta.append({"source_ref": source_ref, "events_seen": len(events)})
        r.variant(rng, "query", {"role": "query", "path": ("queries", qi, "pos")})
        r.words("key", key, "->")
        r.add(ans, {"role": "answer", "path": ("queries", qi, "answer_pos")})
        r.words(".")
        source_by_key[key] = ("query", qi)

    if events:
        add_query(events[-1]["key"])
    else:
        add_query(rng.choice(keys))
    add_query(rng.choice(keys))
    for _ in range(rng.randint(0, 3)):
        r.bg(rng, rng.randint(0, 3))
        add_query(rng.choice(keys))

    rec = {
        "family": "F3",
        "split": split,
        "registry_ids": {"base_rule": rid},
        "composition_depth": composition_depth,
        "token_len": None,
        "active_map_rle": None,
        "events": events,
        "writes": writes,
        "queries": queries,
        "controlled_gaps": controlled_gaps,
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
