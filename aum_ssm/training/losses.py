# AUM-Ø training objective (§16, §18). Seven terms; weights live in AumConfig.
#
#   L = L_LM + L_pressure + L_pred + L_compute + L_consistency + L_precision + L_state
#
# TODO(AUM): implement each term against the silence-block aux outputs (e_t, mu_t, pi_t, J_t,
# sigma trajectory, E_t) and the calibrated benefit target y_t from counterfactual.py.

import torch
import torch.nn.functional as F


def lm_loss(logits, targets):
    """Standard next-token cross-entropy, L_LM (§18)."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def prediction_loss(g_hat, g, lambda_pred):
    """Prediction-head objective L_pred = lambda_P || g_hat - stopgrad(g) ||^2 (§16)."""
    return lambda_pred * (g_hat - g.detach()).pow(2).mean()


def pressure_loss(pi_t, y_t):
    """L_pressure = (pi_t - stopgrad(y_t))^2 against the fixed calibrated target (§17)."""
    return (pi_t - y_t.detach()).pow(2).mean()


def compute_loss(expected_J, lambda_compute):
    """L_compute = lambda_C E[J_t] (§18)."""
    return lambda_compute * expected_J.mean()


def precision_loss(mu_t, lambda_precision):
    """L_precision = lambda_mu ||mu_t||_1 (§18)."""
    return lambda_precision * mu_t.abs().sum(-1).mean()


def state_loss(S_t, lambda_state):
    """L_state = lambda_S ||S_t||^2 (§18)."""
    return lambda_state * S_t.pow(2).mean()


def consistency_loss(E_traj, lambda_consistency):
    """L_consistency = lambda_E sum_j max(0, E_t(sigma^{j+1}) - E_t(sigma^j)) (§18).

    E_traj: tensor (..., J+1) of consistency-functional values along the silent trajectory.
    """
    diffs = E_traj[..., 1:] - E_traj[..., :-1]
    return lambda_consistency * diffs.clamp_min(0).sum(-1).mean()


def total_loss(parts, active_terms):
    """Sum the per-term losses (already lambda-weighted) named in active_terms (§18, §20).

    parts: dict term-name -> scalar tensor. Returns (total_tensor, {term: float}).
    """
    used = {k: parts[k] for k in active_terms if k in parts}
    total = sum(used.values()) if used else torch.zeros(())
    return total, {k: float(v.detach()) for k, v in used.items()}
