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
import torch.utils.checkpoint


def lm_loss(logits, targets):
    """Plain next-token cross-entropy (the silence-ablated / evidence-core path)."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def _fused_kernels(device):
    """The fused CE kernel package for this device (kernels.metal on MPS, kernels.triton on
    CUDA — identical fused_linear_cross_entropy_ce contracts), or None if unavailable."""
    try:
        if device.type == "mps":
            import kernels.metal as km
            return km
        if device.type == "cuda":
            import kernels.triton as km
            return km
    except Exception:
        pass
    return None


class _FusedMixtureCE(torch.autograd.Function):
    """The §8 mixture LM loss via the fused-linear-CE kernels: sum_j w_j*ce_j averaged over
    positions, WITHOUT materializing the (B,L,J+1,V) logits (Liger-style row chunks). Grads for
    the candidate outputs, the halting weights, and the tied classifier weight are computed in
    the forward pass and scaled by the upstream gradient in backward."""

    @staticmethod
    def forward(ctx, o_flat, w_flat, weight, targets_flat, divisor):
        km = _fused_kernels(o_flat.device)
        loss, dh, dW, ce = km.fused_linear_cross_entropy_ce(
            o_flat.detach(), weight.detach(), targets_flat, row_weight=w_flat.detach(),
            divisor=divisor)
        ctx.save_for_backward(dh, dW, ce)
        ctx.divisor = divisor
        return loss

    @staticmethod
    def backward(ctx, g):
        dh, dW, ce = ctx.saved_tensors
        return g * dh, g * ce / ctx.divisor, g * dW, None, None


def lm_mixture_loss(o_stack, w, lm_head, targets):
    """§8/§10: L_LM = sum_j w_j * [-log p^{(j)}(x_{t+1})] — the loss mixture over per-candidate
    outputs. Halting mixes LOSSES, never states; the halting head gets gradient through w.

    o_stack: (B, L, J+1, d) per-candidate outputs at the positions predicting `targets` (B, L).
    w: (B, L, J+1) halting weights. lm_head: the tied classifier (nn.Linear or callable).

    With the fused kernels available (Metal on MPS, Triton on CUDA), uses the fused-linear-CE
    path — the (B,L,J+1,V) logits (2.4 GB at the reference shapes) are never materialized.
    Everywhere else the fallback runs in GRADIENT-CHECKPOINTED row chunks: each chunk's logits
    are recomputed on backward, so peak memory is one (chunk, V) slab instead of the full
    (B*L*(J+1), V) tensor (25.6 GB fp32 at B=2 reference shapes).
    """
    B, L, J1, d = o_stack.shape
    weight = getattr(lm_head, "weight", None)
    if (weight is not None and getattr(lm_head, "bias", None) is None
            and _fused_kernels(o_stack.device) is not None):
        t_flat = targets.unsqueeze(-1).expand(B, L, J1).reshape(-1)
        return _FusedMixtureCE.apply(o_stack.reshape(-1, d), w.reshape(-1), weight, t_flat,
                                     B * L)
    o_flat = o_stack.reshape(-1, d)
    w_flat = w.reshape(-1)
    t_flat = targets.unsqueeze(-1).expand(B, L, J1).reshape(-1)

    def chunk_ce(o_c, w_c, t_c):
        ce = F.cross_entropy(lm_head(o_c).float(), t_c, reduction="none")
        return (w_c * ce).sum()

    rows = o_flat.shape[0]
    chunk = 4096
    total = torch.zeros((), device=o_flat.device, dtype=torch.float32)
    for c0 in range(0, rows, chunk):
        args = (o_flat[c0:c0 + chunk], w_flat[c0:c0 + chunk], t_flat[c0:c0 + chunk])
        if torch.is_grad_enabled() and (o_flat.requires_grad or w_flat.requires_grad):
            total = total + torch.utils.checkpoint.checkpoint(chunk_ce, *args,
                                                              use_reentrant=False)
        else:
            total = total + chunk_ce(*args)
    return total / (B * L)


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
