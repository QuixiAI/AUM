# AUM-Ø counterfactual silence benefit (v6 §11). The supervision signal for integration pressure.
#
# b_t = l_0 - l_K: no-silence vs FIXED-DEPTH-K silence, with paired determinism (shared batch,
# teacher-forced continuation, no dropout in the measurement). Policy-independent label
# (committed): the with-silence branch always runs forced depth K, never the live halting policy —
# the halting head consumes pi_t, so a policy-dependent label would make pi regress onto a target
# that moves with pi. The label is stop-gradient; the pressure target is the fixed monotone
# transform y = log(1 + max(b,0)/beta) in nats (never batch-normalized).

import math

import torch
import torch.nn.functional as F


def per_token_ce(logits, input_ids):
    """Per-position next-token cross-entropy (B, L-1), no reduction."""
    B, L, V = logits.shape
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V), input_ids[:, 1:].reshape(-1),
                         reduction="none")
    return ce.reshape(B, L - 1)


def calibrated_target(b_t, beta=0.02):
    """y_t = log(1 + max(b_t, 0) / beta)  — fixed calibrated target, NOT batch-normalized (§17)."""
    return torch.log1p(b_t.clamp_min(0) / beta)


def one_step_benefit(loss_no_silence, loss_with_silence):
    """b_t^(1) = l_0 - l_J (§17). Both are stop-gradient scalars per token."""
    return (loss_no_silence - loss_with_silence).detach()


def rollout_benefit(model, input_ids, beta=0.02, ablation=None, K=None):
    """Per-position counterfactual silence benefit b_t = l_0 - l_K and target y_t (§11).

    Policy-independent + paired: the with-silence branch runs FIXED depth K (default j_max) via
    forced_depth — never the live halting policy — and both branches share the same inputs and
    dropout-free numerical path, so the only difference is the K silent steps. Returns
    (b, y, aux, result_K) where b (B,L-1) is stop-gradient, y = calibrated_target(b), and
    aux/result_K are the fixed-K forward (reused for the stage-2 LM loss: one graph, not two).

    NOTE: this is the practical all-at-K-vs-all-off, per-position label. The exact §11 policy
    fires silence ONCE at t with downstream frozen off; that per-t rollout is a later refinement.
    """
    if K is None:
        K = model.backbone.silence.j_max
    result_K, aux = model(input_ids, return_aux=True, ablation=ablation, forced_depth=K)
    l_K = per_token_ce(result_K.logits, input_ids)
    was_on = model.backbone.silence_enabled
    model.backbone.silence_enabled = False
    with torch.no_grad():
        l_0 = per_token_ce(model(input_ids).logits, input_ids)
    model.backbone.silence_enabled = was_on
    b = (l_0 - l_K.detach())                     # (B, L-1), stop-gradient benefit label
    return b, calibrated_target(b, beta), aux, result_K
