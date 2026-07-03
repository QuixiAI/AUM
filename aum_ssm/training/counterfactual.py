# AUM-Ø counterfactual silence benefit (v6 §11). The supervision signal for integration pressure.
#
# b_t = l_0 - l_K: no-silence vs FIXED-DEPTH-K silence, with paired determinism (shared batch,
# teacher-forced continuation, no dropout in the measurement — both label branches run under
# no_grad). Policy-independent label (committed): the with-silence branch always runs forced depth
# K, never the live halting policy — the halting head consumes pi_t, so a policy-dependent label
# would make pi regress onto a target that moves with pi. The residual policy-gap is MEASURED, not
# ignored: on_policy_benefit() is the §11 gauge, reported beside the fixed-K correlation.
# The pressure target is the fixed monotone transform y = log(1 + max(b,0)/beta) in nats (never
# batch-normalized — that would destroy the scale that makes delta and lambda_C interpretable).

import torch
import torch.nn.functional as F


def per_token_ce(logits, input_ids, chunk=4096):
    """Per-position next-token cross-entropy (B, L-1), no reduction.

    Chunked over rows: a single F.cross_entropy call materializes the full fp32
    (B*(L-1), V) log-softmax — 3 GiB at (4, 4096, 49152) — and the §11 label branches run
    this up to three times per stage>=2 step, which OOMed 24GB cards the moment the R^2
    gate cleared. Every call site passes detached / no-grad logits, but the chunked form
    stays differentiable anyway."""
    B, L, V = logits.shape
    flat = logits[:, :-1].reshape(-1, V)
    tgt = input_ids[:, 1:].reshape(-1)
    ce = torch.cat([F.cross_entropy(flat[c0:c0 + chunk].float(), tgt[c0:c0 + chunk],
                                    reduction="none")
                    for c0 in range(0, flat.shape[0], chunk)])
    return ce.reshape(B, L - 1)


def calibrated_target(b_t, beta=0.02):
    """y_t = log(1 + max(b_t, 0) / beta) — fixed calibrated target, NOT batch-normalized (§11)."""
    return torch.log1p(b_t.clamp_min(0) / beta)


def rollout_benefit(model, input_ids, beta=0.02, ablation=None, K=None, train_forced_depth=None,
                    raw=None):
    """Fixed-K counterfactual benefit b_t = l_0 - l_K and target y_t (§11), plus the live forward.

    Three branches:
      1. the LIVE forward (with grad; forced_depth=train_forced_depth — K in stage 2, None from
         stage 3 on) whose (result, aux) the trainer reuses for the LM mixture and pi;
      2. l_K: no_grad, forced depth K — the with-silence label branch (never the halting policy);
      3. l_0: no_grad, silence off — the no-silence label branch.
    Stage 2 (train_forced_depth == K) reuses branch 1's logits for l_K: one graph, two measurements.

    Returns (b, y, aux, result): b (B,L-1) stop-gradient, y = calibrated_target(b).

    NOTE: this is the practical all-at-K-vs-all-off, per-position label. The exact §11 policy
    fires silence ONCE at t with downstream frozen off; that per-t rollout is a later refinement.
    """
    raw = raw if raw is not None else model      # unwrapped module for attribute access (DDP)
    if K is None:
        K = raw.backbone.silence.j_max
    result, aux = model(input_ids, return_aux=True, ablation=ablation,
                        forced_depth=train_forced_depth)
    if train_forced_depth == K:
        l_K = per_token_ce(result.logits.detach(), input_ids)
    else:
        with torch.no_grad():
            l_K = per_token_ce(
                model(input_ids, ablation=ablation, forced_depth=K).logits, input_ids)
    was_on = raw.backbone.silence_enabled
    raw.backbone.silence_enabled = False
    with torch.no_grad():
        l_0 = per_token_ce(model(input_ids).logits, input_ids)
    raw.backbone.silence_enabled = was_on
    b = (l_0 - l_K).detach()                     # (B, L-1), stop-gradient benefit label
    return b, calibrated_target(b, beta), aux, result


@torch.no_grad()
def on_policy_benefit(model, input_ids, ablation=None):
    """The §11 on-policy gauge: b under the LIVE halting policy (eval-mode selection).

    Reported beside the fixed-K correlation — it measures the residual policy-gap the fixed-K
    label deliberately ignores. Returns (b_on_policy (B,L-1), aux) with aux from the policy run.
    """
    was_training = model.training
    model.eval()                                  # deterministic j* = min{j: p_j >= delta}
    result, aux = model(input_ids, return_aux=True, ablation=ablation)
    l_pol = per_token_ce(result.logits, input_ids)
    was_on = model.backbone.silence_enabled
    model.backbone.silence_enabled = False
    l_0 = per_token_ce(model(input_ids).logits, input_ids)
    model.backbone.silence_enabled = was_on
    model.train(was_training)
    return l_0 - l_pol, aux
