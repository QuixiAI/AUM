"""Machine-readable SYN-1B sidecar schema and verifier registry.

Metadata is a verified tier, not a trusted one (spec §7). Every emitted field is classified:

  - ``derived``    — a function of the rendered token bytes (optionally + seed/registry). Its
                     owner is a *recomputation verifier* (a member of ``RECOMPUTE_OWNERS``) that
                     re-derives the value along a code path independent of the emitter and
                     asserts ``recomputed == emitted``. A check that only reads the field to
                     bucket/gate instances (a *consumer*) is not a verifier.
  - ``provenance`` — a recorded generation input (seed, corpus_version, placement, ...),
                     re-established by regeneration or the packing/placement contract.
  - ``reported``   — an F5 transfer-probe field, reported not gated (spec §6/§10); its owner is
                     the check that formally records the exemption.

``verifier_registry_mismatches()`` enforces, inside this module: name parity, that every owner
names a *wired* check (``WIRED_VERIFIERS``), and that every ``derived`` field has at least one
``RECOMPUTE_OWNERS`` owner. QA-18 (`schema_admission_audit`) and the run-status coverage gate
(`metadata_recomputation_audit`) enforce the rest against the actual QA run.
"""

# --- Canonical verifier ids -------------------------------------------------------------------
# Base ``name`` ids of QA checks actually wired into ``qa.run_qa`` (split-suffixed names such as
# ``window_integrity_train`` share the base id ``window_integrity``). An owner token that is not
# in this set names a check that does not run, and is build-blocking.
WIRED_VERIFIERS = frozenset({
    "tokenizer", "manifest_tokenizer", "packing_policy", "window_integrity",
    "instance_task_density", "offset_contract", "instance_span_roundtrip",
    "controlled_distribution", "marker", "dedup", "labels", "split_integrity",
    "schema_completeness", "schema_admission", "query_position_contract",
    "distractor_surface", "f3_age", "filler_scaffolding", "train_eval_symbol_exclusion",
    "f1_rule_consistency", "minimal_suffix", "difficulty_strata",
    "composition_depth_recompute", "age_flag_recompute", "provenance_audit",
    "metadata_recomputation",
})

# Owners that re-derive their field from the bytes (or from bytes-verified sibling fields) along
# a path independent of the emitter and assert equality. A ``derived`` field must have one.
RECOMPUTE_OWNERS = frozenset({
    "window_integrity", "offset_contract", "query_position_contract", "distractor_surface",
    "filler_scaffolding", "minimal_suffix", "f1_rule_consistency", "controlled_distribution",
    "instance_task_density", "composition_depth_recompute", "age_flag_recompute",
    "train_eval_symbol_exclusion",
})

REQUIRED_RECORD_FIELDS = {
    "instance_id", "corpus_version", "family", "split", "shard", "window_index",
    "start_offset", "registry_ids", "composition_depth", "token_len",
    "task_token_count", "filler_token_count", "controlled_gap_tokens",
    "density_denominator", "controlled_gap_adjusted_task_fraction", "events",
    "writes", "queries", "distractors", "label_positions", "token_hash",
}

RECORD_FIELD_VERIFIERS = {
    "active_map_rle": "f1_rule_consistency/minimal_suffix",
    "active_param_rle": "minimal_suffix",  # reported (F5 exempt); minimal_suffix records the skip
    "composition_depth": "composition_depth_recompute",
    "controlled_gap_adjusted_task_fraction": "instance_task_density",
    "controlled_gap_rle": "controlled_distribution",
    "controlled_gap_tokens": "controlled_distribution/instance_task_density",
    "controlled_gaps": "controlled_distribution",
    "corpus_version": "manifest_tokenizer/provenance_audit",
    "demonstrations": "minimal_suffix",
    "density_denominator": "instance_task_density",
    "distractors": "distractor_surface",
    "events": "minimal_suffix/f1_rule_consistency",
    "family": "provenance_audit",
    "filler_rle": "filler_scaffolding",
    "filler_token_count": "instance_task_density",
    "instance_id": "provenance_audit",
    "label_positions": "offset_contract",
    "pseudo_event_pos": "distractor_surface",
    "queries": "query_position_contract/minimal_suffix",
    "registry_ids": "provenance_audit",
    "seed": "provenance_audit",
    "shard": "window_integrity",
    "split": "split_integrity/provenance_audit",
    "start_offset": "window_integrity",
    "task_fraction": "instance_task_density",
    "task_token_count": "instance_task_density",
    "text_hash": "provenance_audit",
    "token_hash": "window_integrity",
    "token_len": "window_integrity",
    "window_index": "window_integrity",
    "writes": "minimal_suffix",
}

COMMON_QUERY_FIELDS = {
    "pos", "answer_pos", "answer", "prediction_anchor", "target_age_bin",
    "evidence_positions", "minimal_sufficient_suffix_len", "composition_depth",
    "hard_suffix",
}

COMMON_QUERY_FIELD_VERIFIERS = {
    "answer": "query_position_contract/minimal_suffix",
    "answer_pos": "query_position_contract",
    "composition_depth": "composition_depth_recompute",
    "evidence_positions": "minimal_suffix",
    "hard_suffix": "minimal_suffix",
    "minimal_sufficient_suffix_len": "minimal_suffix",
    "pos": "offset_contract/query_position_contract",
    "prediction_anchor": "query_position_contract",
    "target_age_bin": "controlled_distribution",
}

FAMILY_QUERY_FIELDS = {
    "F1": {
        "key", "age_from_reversal", "age_from_original_statement",
        "mentioned_in_latest_correction",
    },
    "F2": {
        "entity", "age_from_swap", "age_from_binding_statement",
        "entity_was_swapped", "entity_ever_swapped",
    },
    "F3": {
        "key", "age_write_to_query", "age_correction_to_query", "mode",
    },
    "F4": {
        "key",
    },
    "F5": {
        "age_from_event",
    },
}

FAMILY_QUERY_FIELD_VERIFIERS = {
    "F1": {
        "age_from_original_statement": "age_flag_recompute",
        "age_from_reversal": "age_flag_recompute",
        "key": "query_position_contract/minimal_suffix",
        "mentioned_in_latest_correction": "age_flag_recompute/f1_rule_consistency",
    },
    "F2": {
        "age_from_binding_statement": "age_flag_recompute",
        "age_from_swap": "age_flag_recompute",
        "entity": "query_position_contract/minimal_suffix",
        "entity_ever_swapped": "age_flag_recompute",
        "entity_was_swapped": "age_flag_recompute",
    },
    "F3": {
        "age_correction_to_query": "age_flag_recompute",
        "age_write_to_query": "age_flag_recompute/f3_age",
        "key": "query_position_contract/minimal_suffix",
        "mode": "age_flag_recompute",
    },
    "F4": {
        "key": "query_position_contract/minimal_suffix",
    },
    "F5": {
        "age_from_event": "minimal_suffix",  # reported (F5 exempt); minimal_suffix records skip
    },
}

WRITE_FIELDS = {
    "F1": {"key", "pos", "value"},
    "F2": {"entity", "key", "pos", "value"},
    "F3": {"key", "pos", "value"},
    "F4": {"key", "pos", "value"},
    "F5": set(),
}

WRITE_FIELD_VERIFIERS = {
    "entity": "minimal_suffix",
    "key": "minimal_suffix",
    "pos": "offset_contract/minimal_suffix",
    "value": "minimal_suffix",
}

EVENT_FIELDS = {
    "F1": {
        "changed_symbols", "mentioned_symbols", "pos", "restatement", "source_state",
        "target_state", "transition_id", "type",
    },
    "F2": {"event_index", "pos", "swapped_entities", "type"},
    "F3": {"event_index", "key", "new", "old", "pos", "type"},
    "F4": set(),
    "F5": {"new", "old", "pos", "type"},
}

EVENT_FIELD_VERIFIERS = {
    "changed_symbols": "f1_rule_consistency",
    "event_index": "minimal_suffix",
    "key": "minimal_suffix",
    "mentioned_symbols": "f1_rule_consistency",
    "new": "minimal_suffix",
    "old": "minimal_suffix",
    "pos": "offset_contract/minimal_suffix",
    "restatement": "f1_rule_consistency",
    "source_state": "f1_rule_consistency",
    "swapped_entities": "minimal_suffix",
    "target_state": "f1_rule_consistency",
    "transition_id": "provenance_audit",
    "type": "minimal_suffix",
}

DISTRACTOR_FIELDS = {"mimics_family", "pos", "variant"}
DISTRACTOR_FIELD_VERIFIERS = {
    "mimics_family": "distractor_surface",
    "pos": "distractor_surface/offset_contract",
    "variant": "distractor_surface",
}

CONTROLLED_GAP_FIELDS = {"kind", "realized", "target", "target_age_bin"}
CONTROLLED_GAP_FIELD_VERIFIERS = {
    "kind": "controlled_distribution",
    "realized": "controlled_distribution",
    "target": "controlled_distribution",
    "target_age_bin": "controlled_distribution",
}

DEMONSTRATION_FIELDS = {"answer", "answer_pos", "key", "pos"}
DEMONSTRATION_FIELD_VERIFIERS = {
    "answer": "minimal_suffix",
    "answer_pos": "query_position_contract",
    "key": "minimal_suffix",
    "pos": "offset_contract/minimal_suffix",
}

LABEL_POSITION_FIELDS = {"decoded", "expected", "pos", "role"}
LABEL_POSITION_FIELD_VERIFIERS = {
    "decoded": "offset_contract",
    "expected": "offset_contract",
    "pos": "offset_contract",
    "role": "offset_contract",
}

# Field class overrides. Anything not listed here is ``derived`` and must be covered, per QA
# run, by a recomputation verifier's ``verified_fields`` (or be on RECOMPUTE_PENDING). Keys are
# ``(scope, field)`` where scope matches the maps above ("record", "query_common",
# "query_F1"..., "event", "write", "demonstration", ...).
PROVENANCE_FIELDS = {
    ("record", "corpus_version"), ("record", "instance_id"), ("record", "registry_ids"),
    ("record", "seed"), ("record", "shard"), ("record", "window_index"),
    ("record", "start_offset"), ("record", "text_hash"), ("record", "family"),
    ("record", "split"), ("record", "token_hash"),
    ("event", "transition_id"),
}
REPORTED_FIELDS = {
    ("record", "active_param_rle"),
    ("query_F5", "age_from_event"),
}
# Container/list record fields. Their integrity is carried by the verifiers that traverse their
# elements (the nested-scope fields) plus offset/roundtrip placement; they are not themselves a
# scalar to recompute. Covered iff a traversing owner ran and passed.
STRUCTURAL_FIELDS = {
    ("record", "queries"), ("record", "events"), ("record", "writes"),
    ("record", "demonstrations"), ("record", "distractors"), ("record", "label_positions"),
    ("record", "controlled_gaps"), ("record", "active_map_rle"), ("record", "filler_rle"),
    ("record", "controlled_gap_rle"),
}
# The honest backlog: ``derived`` fields with NO independent recompute yet. These are surface
# *content* values that require a second, emitter-independent text decoder (the never-built
# QA-4 replay-content check), plus a few low-value tags. Enforced as an allowlist by
# ``metadata_recomputation_audit``: a NEW derived field that is not covered and not listed here
# is build-blocking, so nothing slips in unverified. Shrinks as decoders are added.
RECOMPUTE_PENDING = {
    ("query_common", "target_age_bin"),
    ("write", "key"), ("write", "value"), ("write", "entity"),
    ("event", "key"), ("event", "new"), ("event", "old"), ("event", "swapped_entities"),
    ("event", "type"), ("event", "event_index"),
    ("event", "source_state"), ("event", "target_state"),
    ("demonstration", "answer"), ("demonstration", "key"),
    ("label_position", "role"),
}


def field_kind(scope: str, field: str) -> str:
    if (scope, field) in PROVENANCE_FIELDS:
        return "provenance"
    if (scope, field) in REPORTED_FIELDS:
        return "reported"
    if (scope, field) in STRUCTURAL_FIELDS:
        return "structural"
    return "derived"


MINIMAL_SUFFIX_EXEMPT_FAMILIES = {
    "F5": (
        "F5 is eval-only and its minimal sufficient evidence is recurrence-parameter "
        "inference from the modular stream, not a finite symbolic write/correction replay."
    ),
}


def required_query_fields(family: str) -> set[str]:
    return set(COMMON_QUERY_FIELDS) | set(FAMILY_QUERY_FIELDS.get(family, set()))


def allowed_record_fields(_family: str) -> set[str]:
    return set(RECORD_FIELD_VERIFIERS)


def allowed_query_fields(family: str) -> set[str]:
    return set(COMMON_QUERY_FIELD_VERIFIERS) | set(FAMILY_QUERY_FIELD_VERIFIERS.get(family, {}))


def verifier_registry_summary() -> dict:
    return {
        "record": RECORD_FIELD_VERIFIERS,
        "query_common": COMMON_QUERY_FIELD_VERIFIERS,
        "query_family": FAMILY_QUERY_FIELD_VERIFIERS,
        "write": WRITE_FIELD_VERIFIERS,
        "event": EVENT_FIELD_VERIFIERS,
        "distractor": DISTRACTOR_FIELD_VERIFIERS,
        "controlled_gap": CONTROLLED_GAP_FIELD_VERIFIERS,
        "demonstration": DEMONSTRATION_FIELD_VERIFIERS,
        "label_position": LABEL_POSITION_FIELD_VERIFIERS,
    }


def _scoped_verifier_items():
    """Yield (scope, field, owner_string) over every registered field."""
    for field, owner in RECORD_FIELD_VERIFIERS.items():
        yield "record", field, owner
    for field, owner in COMMON_QUERY_FIELD_VERIFIERS.items():
        yield "query_common", field, owner
    for family, owners in FAMILY_QUERY_FIELD_VERIFIERS.items():
        for field, owner in owners.items():
            yield f"query_{family}", field, owner
    for field, owner in WRITE_FIELD_VERIFIERS.items():
        yield "write", field, owner
    for field, owner in EVENT_FIELD_VERIFIERS.items():
        yield "event", field, owner
    for field, owner in DISTRACTOR_FIELD_VERIFIERS.items():
        yield "distractor", field, owner
    for field, owner in CONTROLLED_GAP_FIELD_VERIFIERS.items():
        yield "controlled_gap", field, owner
    for field, owner in DEMONSTRATION_FIELD_VERIFIERS.items():
        yield "demonstration", field, owner
    for field, owner in LABEL_POSITION_FIELD_VERIFIERS.items():
        yield "label_position", field, owner


def owners_for(owner_string: str) -> list[str]:
    return [tok for tok in owner_string.split("/") if tok]


def field_owner_index() -> dict:
    """Map ``(scope, field)`` -> {kind, owners} over the whole registry."""
    index = {}
    for scope, field, owner in _scoped_verifier_items():
        index[(scope, field)] = {"kind": field_kind(scope, field),
                                  "owners": owners_for(owner)}
    return index


def verifier_registry_mismatches() -> list[dict]:
    """Static registry drift, independent of any QA run.

    Catches: (1) schema/verifier name parity; (2) an owner that names a check not wired into
    the run (``unknown_owner``); (3) a ``derived`` field with no recomputation owner
    (``no_recompute_owner`` — the consumer-only-metadata bug that shipped
    ``composition_depth``).
    """
    bad = []

    def compare(name: str, fields: set[str], verifiers: dict[str, str], *, allow_extra=False):
        missing = sorted(set(fields) - set(verifiers))
        extra = [] if allow_extra else sorted(set(verifiers) - set(fields))
        if missing or extra:
            bad.append({"schema": name, "missing_verifier": missing, "orphan_verifier": extra})

    compare("record_required", REQUIRED_RECORD_FIELDS, RECORD_FIELD_VERIFIERS, allow_extra=True)
    compare("query_common", COMMON_QUERY_FIELDS, COMMON_QUERY_FIELD_VERIFIERS)
    for family, fields in FAMILY_QUERY_FIELDS.items():
        compare(f"query_{family}", fields, FAMILY_QUERY_FIELD_VERIFIERS.get(family, {}))
    compare("write", set().union(*WRITE_FIELDS.values()), WRITE_FIELD_VERIFIERS)
    compare("event", set().union(*EVENT_FIELDS.values()), EVENT_FIELD_VERIFIERS)
    compare("distractor", DISTRACTOR_FIELDS, DISTRACTOR_FIELD_VERIFIERS)
    compare("controlled_gap", CONTROLLED_GAP_FIELDS, CONTROLLED_GAP_FIELD_VERIFIERS)
    compare("demonstration", DEMONSTRATION_FIELDS, DEMONSTRATION_FIELD_VERIFIERS)
    compare("label_position", LABEL_POSITION_FIELDS, LABEL_POSITION_FIELD_VERIFIERS)

    for scope, field, owner in _scoped_verifier_items():
        tokens = owners_for(owner)
        unknown = sorted(set(tokens) - WIRED_VERIFIERS)
        if unknown:
            bad.append({"scope": scope, "field": field, "unknown_owner": unknown,
                        "owner": owner})
        if field_kind(scope, field) == "derived" and not (set(tokens) & RECOMPUTE_OWNERS):
            bad.append({"scope": scope, "field": field, "no_recompute_owner": tokens,
                        "kind": "derived", "owner": owner})
    return bad
