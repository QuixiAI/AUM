# AUM-Ø synthetic task families (§22). Each yields sequences with a known latent hypothesis and a
# controlled evidence-age axis, so the recency gradient corr(b_t, evidence-age) < 0 can be measured.
#
# TODO(AUM): implement generators returning (input_ids, targets, meta) where meta carries the
# event position and evidence-age for the §22 analyses and the §23 gate.


def branch_reversal(rng, length, **kwargs):
    """A rule holds, a reversal token flips it (§22.1). Recent trigger."""
    raise NotImplementedError


def latent_binding_swap(rng, length, **kwargs):
    """'A=red, B=blue, C=green ... Correction: A and C were swapped. What color is A?' (§22.2).

    Same evidence/hypothesis structure as reversal, different surface form, so the register
    cannot pass by detecting a literal 'reversal' token.
    """
    raise NotImplementedError


def delayed_correction(rng, length, **kwargs):
    """Old evidence must be reinterpreted (§22.3). The evidence-age axis; base v5.3 helps less."""
    raise NotImplementedError


def flat_null(rng, length, **kwargs):
    """No interpretive events (§22.4). Registered null: pi ~ 0, E[J] -> 0."""
    raise NotImplementedError


TASKS = {
    "branch_reversal": branch_reversal,
    "latent_binding_swap": latent_binding_swap,
    "delayed_correction": delayed_correction,
    "flat_null": flat_null,
}
