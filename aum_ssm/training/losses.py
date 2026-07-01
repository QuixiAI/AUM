# AUM-Ø training objective (v6 §10). Seven terms; weights live in AumConfig.
#
#   L = L_LM + L_pressure + L_pred + lambda_C E[J] + L_consistency
#       + lambda_mu sum_l ||mu^l||_1 + lambda_S ||S||^2
#
# Two deliberate asymmetries (both anti-degeneracy, §10): the l1 sparsity applies to PER-LAYER
# precision only — the global mu may not be starved to zero to satisfy a regularizer — and the
# consistency functional consumes detached precision (enforced inside SilenceBlock._consistency).

import torch
import torch.nn.functional as F


def lm_loss(logits, targets):
    """Plain next-token cross-entropy (the silence-ablated / evidence-core path)."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def lm_mixture_loss(o_stack, w, lm_head, targets):
    """§8/§10: L_LM = sum_j w_j * [-log p^{(j)}(x_{t+1})] — the loss mixture over per-candidate
    outputs. Halting mixes LOSSES, never states; the halting head gets gradient through w.

    o_stack: (B, L, J+1, d) per-candidate outputs at the positions predicting `targets` (B, L).
    w: (B, L, J+1) halting weights. lm_head: callable d -> vocab logits (the tied classifier).
    """
    logits = lm_head(o_stack)                                    # (B, L, J+1, V)
    B, L, J1, V = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, V),
                         targets.unsqueeze(-1).expand(B, L, J1).reshape(-1),
                         reduction="none").reshape(B, L, J1)
    return (w * ce).sum(-1).mean()


def prediction_loss(g_hat, g, lambda_pred):
    """L_pred = lambda_P || g_hat - stopgrad(g) ||^2 (§10)."""
    return lambda_pred * (g_hat - g.detach()).pow(2).mean()


def pressure_loss(pi_t, y_t, mask=None):
    """L_pressure = (pi_t - stopgrad(y_t))^2 against the fixed calibrated target (§11).

    mask (optional, same shape): restrict to the §11 exploration subset — the tokens where the
    fixed-K label was actually recorded.
    """
    se = (pi_t - y_t.detach()).pow(2)
    if mask is not None:
        return (se * mask).sum() / mask.sum().clamp_min(1.0)
    return se.mean()


def compute_loss(expected_J, lambda_compute):
    """lambda_C * E[J_t] (§10) — the collapse knob, ramped from ~0 in stage 3 (§12)."""
    return lambda_compute * expected_J.mean()


def precision_loss(mu_layers, lambda_precision):
    """lambda_mu * sum_l ||mu_t^l||_1 over the PER-LAYER precision fields only (§10).

    mu_layers: a per-layer tensor list (or one tensor). Never pass the global mu — starving it to
    zero to satisfy the regularizer is exactly the v5.3 degenerate basin v6 closes.
    """
    if torch.is_tensor(mu_layers):
        mu_layers = [mu_layers]
    return lambda_precision * sum(m.abs().sum(-1).mean() for m in mu_layers)


def state_loss(S_t, lambda_state):
    """L_state = lambda_S ||S_t||^2 (§10)."""
    return lambda_state * S_t.pow(2).mean()


def consistency_loss(E_traj, lambda_consistency):
    """L_consistency = lambda_E sum_j max(0, E(sigma^{j+1}) - E(sigma^j)) (§10).

    E_traj: (..., J+1) consistency-functional values along the silent trajectory.
    """
    diffs = E_traj[..., 1:] - E_traj[..., :-1]
    return lambda_consistency * diffs.clamp_min(0).sum(-1).mean()


def total_loss(parts, active_terms):
    """Sum the per-term losses (already lambda-weighted) named in active_terms (§10, §12).

    parts: dict term-name -> scalar tensor. Returns (total_tensor, {term: float}).
    """
    used = {k: parts[k] for k in active_terms if k in parts}
    total = sum(used.values()) if used else torch.zeros(())
    return total, {k: float(v.detach()) for k, v in used.items()}
