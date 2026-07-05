"""F1 branch-reversal generator."""

from __future__ import annotations

from train.syn import MAX_ATTENTION_WINDOW_PLANNED
from train.syn.render import Renderer


def _base_mapping(rule, working):
    perm = rule.data.get("base_perm", rule.data.get("perm"))
    return {working[i]: working[perm[i]] for i in range(8)}


def _apply_transition(current, working, rule, restatement):
    if restatement == "full":
        changed_indices = list(range(8))
        delta = rule.data["delta8"]
    else:
        changed_indices = list(rule.data["changed_order"][:3])
        delta = rule.data["delta3"]
    old_images = [current[working[i]] for i in changed_indices]
    new_map = dict(current)
    for j, idx in enumerate(changed_indices):
        new_map[working[idx]] = old_images[delta[j]]
    changed_symbols = [working[i] for i in changed_indices]
    return new_map, changed_symbols


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
        if kind == "demo":
            return rec["demonstrations"][key_or_index]["pos"]
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
        q["age_from_original_statement"] = int(q["answer_pos"] - write_pos[q["key"]])
        q["age_from_reversal"] = (
            None if events_seen == 0 else int(q["answer_pos"] - event_positions[events_seen - 1])
        )
        q["mentioned_in_latest_correction"] = (
            False if events_seen == 0
            else q["key"] in rec["events"][events_seen - 1]["mentioned_symbols"]
        )


def generate(alpha, registry, rng, split, instance_index, target_len, age_bin=0):
    pool = list(alpha.sigma_train if split == "train" else alpha.sigma_eval + alpha.sigma_train)
    working = rng.sample(pool, 8)
    ids_for_split = list(registry.ids_for_split(split))
    base_id = rng.choice(ids_for_split)
    composition_depth = 2 if instance_index % 3 == 0 else 1
    transition_ids = rng.sample(ids_for_split, composition_depth)

    map_current = _base_mapping(registry.rule_for("F1", base_id), working)
    state_maps = [dict(map_current)]
    source_by_key = {key: ("write", key) for key in working}

    r = Renderer(alpha)
    r.variant(rng, "rule")
    writes = []
    demonstrations = []
    queries = []
    query_meta = []
    controlled_gaps = []

    def add_demo(key, answer):
        di = len(demonstrations)
        demonstrations.append({"pos": None, "answer_pos": None, "key": key, "answer": answer})
        r.add(key, {"role": "demo", "path": ("demonstrations", di, "pos")})
        r.words("->")
        r.add(answer, {"role": "demo_answer", "path": ("demonstrations", di, "answer_pos")})
        r.words(".")
        source_by_key[key] = ("demo", di)

    def add_query(key, answer, events_seen, update_evidence=True, hard_suffix=False):
        qi = len(queries)
        source_ref = source_by_key[key]
        queries.append({
            "pos": None,
            "answer_pos": None,
            "key": key,
            "answer": answer,
            "prediction_anchor": "->",
            "age_from_reversal": None,
            "age_from_original_statement": None,
            "target_age_bin": age_bin,
            "composition_depth": composition_depth,
            "mentioned_in_latest_correction": None,
            "evidence_positions": None,
            "minimal_sufficient_suffix_len": None,
            "hard_suffix": bool(hard_suffix),
        })
        query_meta.append({"source_ref": source_ref, "events_seen": events_seen})
        r.variant(rng, "query", {"role": "query", "path": ("queries", qi, "pos")})
        r.words(key, "->")
        r.add(answer, {"role": "answer", "path": ("queries", qi, "answer_pos")})
        r.words(".")
        if update_evidence:
            source_by_key[key] = ("query", qi)

    for key, value in map_current.items():
        wi = len(writes)
        writes.append({"pos": None, "key": key, "value": value})
        r.add(key, {"role": "write", "path": ("writes", wi, "pos")})
        r.words("means", value, ".")

    for demo_i in range(rng.randint(8, 12)):
        r.bg(rng, rng.randint(0, 2))
        key = rng.choice(working)
        if demo_i < 2:
            add_query(key, map_current[key], events_seen=0)
        else:
            add_demo(key, map_current[key])

    events = []
    latest_changed = []
    controlled_gap_used = False
    for event_index, transition_id in enumerate(transition_ids):
        r.bg(rng, rng.randint(3, 12))
        restatement = "partial" if (instance_index + event_index) % 4 != 0 else "full"
        transition = registry.rule_for("F1", transition_id)
        next_map, changed_symbols = _apply_transition(map_current, working, transition, restatement)
        latest_changed = changed_symbols
        events.append({
            "type": "reversal",
            "pos": None,
            "transition_id": transition_id,
            "source_state": f"S{event_index}",
            "target_state": f"S{event_index + 1}",
            "restatement": restatement,
            "changed_symbols": changed_symbols,
            "mentioned_symbols": list(changed_symbols),
        })
        r.variant(rng, "correction", {"role": "event", "path": ("events", event_index, "pos")})
        mentioned = working if restatement == "full" else changed_symbols
        for key in mentioned:
            r.words(key, "now", "means", next_map[key], ".")
        map_current = next_map
        state_maps.append(dict(map_current))
        for key in changed_symbols:
            source_by_key[key] = ("event", event_index)

        if restatement == "partial":
            unmentioned = [k for k in working if k not in changed_symbols]
            if unmentioned:
                hard_candidates = [
                    k for k in unmentioned
                    if source_by_key.get(k, (None,))[0] not in {"demo", "query"}
                ]
                key = rng.choice(hard_candidates or unmentioned)
                if controlled_gap_used:
                    gap = rng.randint(3, 12)
                    hard_suffix = False
                    r.bg(rng, gap)
                else:
                    max_gap = target_len - r.token_len() - 96
                    gap = _age_target(age_bin, rng, max_gap)
                    controlled_gaps.append({"kind": "age", "target": gap, "realized": gap,
                                            "target_age_bin": age_bin})
                    r.bg(rng, gap, controlled=True)
                    controlled_gap_used = True
                    hard_suffix = gap > MAX_ATTENTION_WINDOW_PLANNED
                add_query(key, map_current[key], events_seen=event_index + 1,
                          hard_suffix=hard_suffix)
                r.bg(rng, rng.randint(0, 2))
                key2 = rng.choice(unmentioned)
                add_query(key2, map_current[key2], events_seen=event_index + 1)

        if event_index + 1 < composition_depth:
            for _ in range(rng.randint(2, 4)):
                r.bg(rng, rng.randint(0, 2))
                choices = [k for k in working if k not in latest_changed] or working
                key = rng.choice(choices)
                add_demo(key, map_current[key])

    for _ in range(rng.randint(4, 8)):
        r.bg(rng, rng.randint(0, 3))
        key = rng.choice(working)
        add_demo(key, map_current[key])

    total_queries = rng.randint(4, 8)
    post_keys = []
    unmentioned = [k for k in working if k not in latest_changed]
    if events and events[-1]["restatement"] == "partial" and unmentioned:
        post_keys.append(rng.choice(unmentioned))
    while len(queries) + len(post_keys) < total_queries:
        post_keys.append(rng.choice(working))
    for key in post_keys:
        r.bg(rng, rng.randint(0, 3))
        add_query(key, map_current[key], events_seen=composition_depth)

    rec = {
        "family": "F1",
        "split": split,
        "registry_ids": {"base_rule": base_id, "transitions": transition_ids},
        "composition_depth": composition_depth,
        "token_len": None,
        "active_map_rle": None,
        "events": events,
        "writes": writes,
        "demonstrations": demonstrations,
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
