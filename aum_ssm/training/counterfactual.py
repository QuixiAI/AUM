# AUM-Ø counterfactual silence benefit (§17). The supervision signal for integration pressure.
#
# One-step benefit b_t^(1) = l_0 - l_J (no-silence vs with-silence loss), with a short-horizon
# variant b_t^(K). Downstream silence is FROZEN OFF in t+1..t+K for both branches, with paired
# determinism (shared batch, teacher-forced continuation, dropout masks, precision path). The
# benefit label is stop-gradient. The pressure target is a fixed monotone transform in nats.

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


def rollout_benefit(model, input_ids, beta=0.02, ablation=None):
    """Per-position counterfactual silence benefit b_t and calibrated target y_t (§17).

    Paired: the two branches share the same inputs and (dropout-free) numerical path, so the
    only difference is whether the silence block is active. Returns (b, y, aux, result_J) where
    b (B,L-1) = l_0 - l_J is stop-gradient, y = calibrated_target(b), aux/result_J are the
    silence-ON forward (reused for the LM loss so we pay one graph, not two).

    NOTE: this is the practical all-on-vs-all-off, per-position benefit. The exact §17 policy
    fires silence ONCE at t with downstream frozen off; that per-t rollout is a later refinement.
    """
    result_J, aux = model(input_ids, return_aux=True, ablation=ablation)
    l_J = per_token_ce(result_J.logits, input_ids)
    was_on = model.backbone.silence_enabled
    model.backbone.silence_enabled = False
    with torch.no_grad():
        l_0 = per_token_ce(model(input_ids).logits, input_ids)
    model.backbone.silence_enabled = was_on
    b = (l_0 - l_J.detach())                     # (B, L-1), stop-gradient benefit label
    return b, calibrated_target(b, beta), aux, result_J
