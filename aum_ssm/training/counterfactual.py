# AUM-Ø counterfactual silence benefit (§17). The supervision signal for integration pressure.
#
# One-step benefit b_t^(1) = l_0 - l_J (no-silence vs with-silence loss), with a short-horizon
# variant b_t^(K). Downstream silence is FROZEN OFF in t+1..t+K for both branches, with paired
# determinism (shared batch, teacher-forced continuation, dropout masks, precision path). The
# benefit label is stop-gradient. The pressure target is a fixed monotone transform in nats.

import math

import torch


def calibrated_target(b_t, beta=0.02):
    """y_t = log(1 + max(b_t, 0) / beta)  — fixed calibrated target, NOT batch-normalized (§17)."""
    return torch.log1p(b_t.clamp_min(0) / beta)


def one_step_benefit(loss_no_silence, loss_with_silence):
    """b_t^(1) = l_0 - l_J (§17). Both are stop-gradient scalars per token."""
    return (loss_no_silence - loss_with_silence).detach()


def rollout_benefit(model, batch, t, horizon=1, weights=None):
    """Compute b_t^(K) by running the frozen-downstream paired rollout (§17).

    TODO(AUM): run two forward branches that differ ONLY in whether silence fired once at t;
    freeze silence off for t+1..t+K in both; share dropout masks / numerical path (see
    aum_ssm.utils.determinism); return the stop-gradient benefit label.
    """
    raise NotImplementedError("Implement the frozen-downstream paired counterfactual rollout (§17).")
