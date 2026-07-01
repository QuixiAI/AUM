# AUM-Ø synthetic task families (§22). Small-vocab next-token tasks with a known latent
# hypothesis and a controlled evidence-age, so the recency gradient corr(b_t, evidence-age)<0
# and the null (pi~0) can be measured. Each generator returns (input_ids (L,), meta) where meta
# carries event_pos (the token whose next-step prediction benefits from revision) and evidence_age.

import torch

# Token ids (small shared vocab; pad to a multiple of 8)
PAD, BOS, FLIP, QUERY, SWAP = 0, 1, 2, 3, 4
SLOT0 = 5                    # SLOT0, SLOT0+1, SLOT0+2
VAL0 = 8                     # VAL0 .. VAL0+NVAL-1
NVAL = 10
VOCAB_SIZE = 24             # 8 specials/slots + 10 values, padded


def _reverse(v):
    return NVAL - 1 - v


def _latent_rule_seq(rng, length, event_distance):
    """A rule (identity vs reversal) toggled by FLIP; a QUERY asks for transform(last_value, rule)."""
    query_pos, answer_pos = length - 2, length - 1
    flip_pos = max(1, query_pos - event_distance)
    seq = [BOS] + [PAD] * (length - 1)
    r, cur_v = 0, rng.randrange(NVAL)
    for i in range(1, length):
        if i == flip_pos:
            seq[i] = FLIP
            r ^= 1
        elif i == query_pos:
            seq[i] = QUERY
        elif i == answer_pos:
            seq[i] = VAL0 + (cur_v if r == 0 else _reverse(cur_v))
        else:
            cur_v = rng.randrange(NVAL)
            seq[i] = VAL0 + cur_v
    meta = {"event_pos": query_pos, "answer_pos": answer_pos, "evidence_age": event_distance,
            "flip_pos": flip_pos, "rule": r}
    return torch.tensor(seq, dtype=torch.long), meta


def branch_reversal(rng, length, event_distance=2):
    """A rule holds, a recent reversal token flips it (§22.1)."""
    return _latent_rule_seq(rng, length, event_distance)


def delayed_correction(rng, length, event_distance=None):
    """Old evidence must be reinterpreted (§22.3) — the evidence-age axis. FLIP placed far back."""
    if event_distance is None:
        event_distance = max(2, length // 2)
    return _latent_rule_seq(rng, length, event_distance)


def latent_binding_swap(rng, length):
    """A=x,B=y,C=z; a SWAP swaps two slots; QUERY asks slot 0's current value (§22.2).

    Same evidence/hypothesis structure as reversal but a different surface form (no literal
    'reversal' token), so the register cannot pass by detecting a reversal cue.
    """
    slots = [rng.randrange(NVAL) for _ in range(3)]
    seq = [BOS, SLOT0, VAL0 + slots[0], SLOT0 + 1, VAL0 + slots[1], SLOT0 + 2, VAL0 + slots[2]]
    b = rng.randrange(1, 3)                       # swap slot 0 with slot b (so slot 0 may change)
    seq += [SWAP, SLOT0, SLOT0 + b]
    slots[0], slots[b] = slots[b], slots[0]
    swap_end = len(seq)
    query_pos = length - 2
    seq = seq[:query_pos]
    while len(seq) < query_pos:
        seq.append(VAL0 + rng.randrange(NVAL))
    seq += [QUERY, VAL0 + slots[0]]               # answer = slot 0's current value
    ids = torch.tensor(seq[:length], dtype=torch.long)
    meta = {"event_pos": query_pos, "answer_pos": length - 1,
            "evidence_age": max(1, query_pos - swap_end), "rule": slots[0]}
    return ids, meta


def flat_null(rng, length):
    """No interpretive events (§22.4). Registered null: pi ~ 0, E[J] -> 0."""
    seq = [BOS] + [VAL0 + rng.randrange(NVAL) for _ in range(length - 1)]
    return torch.tensor(seq, dtype=torch.long), {"event_pos": None, "evidence_age": None}


TASKS = {
    "branch_reversal": branch_reversal,
    "latent_binding_swap": latent_binding_swap,
    "delayed_correction": delayed_correction,
    "flat_null": flat_null,
}


def make_batch(task_fn, rng, batch_size, length, **kw):
    """Returns (input_ids (B,L) long, list[meta])."""
    ids, metas = [], []
    for _ in range(batch_size):
        i, m = task_fn(rng, length, **kw)
        ids.append(i)
        metas.append(m)
    return torch.stack(ids), metas
