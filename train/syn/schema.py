"""Machine-readable SYN-1B sidecar schema.

The generator emits these canonical fields. Reader code may keep compatibility fallbacks for
older dry runs, but QA-11 validates this schema only.
"""

REQUIRED_RECORD_FIELDS = {
    "instance_id", "corpus_version", "family", "split", "shard", "window_index",
    "start_offset", "registry_ids", "composition_depth", "token_len",
    "task_token_count", "filler_token_count", "controlled_gap_tokens",
    "density_denominator", "controlled_gap_adjusted_task_fraction", "events",
    "writes", "queries", "distractors", "label_positions", "token_hash",
}

COMMON_QUERY_FIELDS = {
    "pos", "answer_pos", "answer", "prediction_anchor", "target_age_bin",
    "evidence_positions", "minimal_sufficient_suffix_len", "composition_depth",
    "hard_suffix",
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

MINIMAL_SUFFIX_EXEMPT_FAMILIES = {
    "F5": (
        "F5 is eval-only and its minimal sufficient evidence is recurrence-parameter "
        "inference from the modular stream, not a finite symbolic write/correction replay."
    ),
}


def required_query_fields(family: str) -> set[str]:
    return set(COMMON_QUERY_FIELDS) | set(FAMILY_QUERY_FIELDS.get(family, set()))
