# AUM-Ø v6 gate harness (§14/§15). Trains config-toggled variants (baselines/ablations/controls)
# on the synthetic tasks and checks the minimum-viable-proof table. Probe and calibration numbers
# are measured on HELD-OUT generators (§13: disjoint value ranges — unseen surface, same latent
# structure). A real pass/fail needs training to convergence; this module provides the machinery.

import random
from dataclasses import dataclass, field

import torch

from aum_ssm.models.config_aum import AumConfig
from aum_ssm.models.aum_lm import AumLMHeadModel
from aum_ssm.training.schedule import Stage, ScheduleConfig
from aum_ssm.training.trainer import AumTrainer
from aum_ssm.training.counterfactual import rollout_benefit, on_policy_benefit
from aum_ssm.training.tasks import synthetic as S
from aum_ssm.training import diagnostics as D
from aum_ssm.utils.generation import InferenceParams


@dataclass
class Variant:
    name: str
    config: dict = field(default_factory=dict)   # AumConfig overrides
    ablation: str = None                          # runtime ablation passed to forward


# The §14 / Appendix-B comparison set — all one architecture under flags (param/compute-matched).
VARIANTS = [
    Variant("full", {"silence_enabled": True}),
    Variant("evidence_core", {"silence_enabled": False}),                 # baseline (silence ablated)
    Variant("top_gru", {"silence_enabled": True, "baseline": "top_gru"}), # adapter baseline (no S read)
    Variant("no_op", {"silence_enabled": True}, "no_op"),
    Variant("no_read", {"silence_enabled": True}, "no_read"),
    Variant("phase_scrambled", {"silence_enabled": True}, "phase_scrambled"),
    Variant("random", {"silence_enabled": True}, "random"),
    Variant("no_pressure", {"silence_enabled": True, "lambda_pressure": 0.0}),
    Variant("no_pred", {"silence_enabled": True, "lambda_pred": 0.0}),
    Variant("entropy_feature", {"silence_enabled": True, "entropy_feature": True}),  # §14 ablation
]


def _base_config(**over):
    base = dict(vocab_size=S.VOCAB_SIZE, n_layer=2, d_intermediate=128)
    base.update(over)
    return AumConfig(**base)


@torch.no_grad()
def answer_accuracy(model, ids, metas, ablation=None):
    """Fraction of interpretive events whose answer token is top-1 predicted at event_pos."""
    logits = model(ids, ablation=ablation).logits
    correct = total = 0
    for b, meta in enumerate(metas):
        ep, ap = meta.get("event_pos"), meta.get("answer_pos")
        if ep is None or ap is None:
            continue
        correct += int(logits[b, ep].argmax(-1) == ids[b, ap])
        total += 1
    return correct / max(total, 1)


def evaluate(model, cfg, ablation, rng, task_fn=S.branch_reversal, batch=16, length=12):
    """HELD-OUT metrics for a trained variant (accuracy, calibration, allocation, decodability,
    sigma-relevance, on-policy gauge, phase velocities). Runs the deterministic eval policy."""
    was_training = model.training
    model.eval()
    ids, metas = S.make_batch(task_fn, rng, batch, length, holdout=True)   # held-out generator (§13)
    acc = answer_accuracy(model, ids, metas, ablation)
    out = {"accuracy": acc, "mean_pi": 0.0, "mean_J": 0.0, "corr_pi_b": 0.0,
           "corr_pi_b_on_policy": float("nan"), "sigma_decode": float("nan"),
           "sigma_relevance": float("nan")}
    if not cfg.silence_enabled:
        model.train(was_training)
        return out
    b, _, aux, _ = rollout_benefit(model, ids, cfg.beta, ablation)         # fixed-K label (§11)
    out["mean_pi"] = float(aux.pi.mean().detach())
    out["mean_J"] = float(aux.expected_J.mean().detach())
    out["corr_pi_b"] = D.corr_pi_benefit(aux.pi.detach(), b)
    b_pol, aux_pol = on_policy_benefit(model, ids, ablation)               # the §11 gauge
    out["corr_pi_b_on_policy"] = D.corr_pi_benefit(aux_pol.pi, b_pol)
    out["phase_velocity"] = D.phase_velocity_stats(aux.phi.detach())
    if cfg.baseline != "top_gru":
        out["sigma_relevance"] = D.sigma_relevance(model, ids)             # §16 bypass check
    # sigma-decode: decode the latent rule from the CARRIED register at the event position (§16)
    ev = torch.tensor([m["event_pos"] for m in metas])
    sig = aux.sigma_star.detach()[torch.arange(len(metas)), ev]  # (B, d_sigma)
    labels = torch.tensor([int(m.get("rule", 0)) % 2 for m in metas])
    if labels.unique().numel() > 1:
        out["sigma_decode"] = D.sigma_decode_probe(sig, labels, n_classes=2)
    model.train(was_training)
    return out


def recency_gradient(model, cfg, rng, ages=(2, 5, 9), batch=16, length=14):
    """§14 registered falsifier, re-registered on ACCUMULATED PHASE DISTANCE: corr(b_t, dphi) < 0,
    with token-age as the secondary covariate. dphi = mean over heads of
    phi[t_reinterpret] - phi[t_evidence]; b_t is the fixed-K benefit at the event."""
    if not cfg.silence_enabled:
        return {"corr_b_dphi": float("nan"), "corr_b_age": float("nan")}
    was_training = model.training
    model.eval()
    bs, dphis, age_x = [], [], []
    for age in ages:
        ids, metas = S.make_batch(S.delayed_correction, rng, batch, length,
                                  event_distance=age, holdout=True)
        b, _, aux, _ = rollout_benefit(model, ids, cfg.beta)
        phi = aux.phi.detach()
        for i, m in enumerate(metas):
            ev = min(m["event_pos"], b.shape[1] - 1)
            bs.append(float(b[i, ev]))
            dphis.append(float((phi[i, m["event_pos"]] - phi[i, m["flip_pos"]]).mean()))
            age_x.append(float(age))
    model.train(was_training)
    t = lambda v: torch.tensor(v, dtype=torch.float32)
    return {"corr_b_dphi": D.corr(t(bs), t(dphis)), "corr_b_age": D.corr(t(bs), t(age_x))}


@torch.no_grad()
def evidence_state_features(model, ids, t):
    """Top-layer S_t after consuming ids[:, :t+1] (prefill state capture), flattened per row."""
    B, T = ids.shape
    ip = InferenceParams(max_seqlen=T, max_batch_size=B)
    ip.key_value_memory_dict = model.allocate_inference_cache(B, T)
    model(ids[:, :t + 1], inference_params=ip)
    top = len(model.backbone.layers) - 1
    S_state = ip.key_value_memory_dict[top]["unfold"][1]         # (B, H, Dv, Dqk)
    return S_state.reshape(B, -1)


def evidence_survival_probe(model, cfg, rng, batch=24, length=14):
    """§16: on old-evidence tasks, is the association still linearly recoverable from S_t at query
    time? Not recoverable -> the evidence DECAYED (alpha story; no read policy can retrieve it).
    Recoverable but unread -> the addressing story stands and temporal search (§17) is the right
    extension. This probe keeps the recency result interpretable."""
    was_training = model.training
    model.eval()
    ids, metas = S.make_batch(S.delayed_correction, rng, batch, length, holdout=True)
    feats = evidence_state_features(model, ids, metas[0]["event_pos"])
    model.train(was_training)
    labels = torch.tensor([int(m["rule"]) % 2 for m in metas])
    if labels.unique().numel() < 2:
        return float("nan")
    return D.sigma_decode_probe(feats, labels, n_classes=2)


def train_variant(variant: Variant, steps, rng, lr=1e-3, batch=16, length=12,
                  task_fns=(S.branch_reversal, S.latent_binding_swap), gate_eta_r2=-1e9):
    cfg = _base_config(**variant.config)
    model = AumLMHeadModel(cfg)
    tr = AumTrainer(model, torch.optim.Adam(model.parameters(), lr), cfg,
                    ScheduleConfig(eta_r2=gate_eta_r2))     # smoke default: gate always clears
    tr.force_ablation = variant.ablation
    for i in range(steps):
        fn = task_fns[i % len(task_fns)]
        ids, _ = S.make_batch(fn, rng, batch, length)
        tr.train_step(ids)
        if i == steps // 3:                                     # advance past the pressure gate midway
            tr.maybe_advance_stage(tr.pred_val_r2(ids))
    return model, cfg, tr


def run_gate(steps=40, seed=0, variants=VARIANTS):
    """Train + evaluate every variant; return metrics and the §14/§15 gate checks (pass/fail)."""
    results = {}
    for v in variants:
        rng = random.Random(seed)
        model, cfg, _ = train_variant(v, steps, rng)
        ev = random.Random(seed + 1)
        m = evaluate(model, cfg, v.ablation, ev)
        m["recency"] = recency_gradient(model, cfg, random.Random(seed + 2))
        if v.name == "full":                                     # §16 evidence-survival probe
            m["evidence_survival"] = evidence_survival_probe(model, cfg, random.Random(seed + 4))
        # null control: pi on flat_null should be ~0 (held-out values)
        nids, _ = S.make_batch(S.flat_null, random.Random(seed + 3), 16, 12, holdout=True)
        if cfg.silence_enabled:
            _, _, naux, _ = rollout_benefit(model, nids, cfg.beta, v.ablation)
            m["null_pi"] = float(naux.pi.mean().detach())
        results[v.name] = m

    f = results.get("full", {})
    checks = {
        "beats_evidence_core": f.get("accuracy", 0) > results.get("evidence_core", {}).get("accuracy", 1),
        "beats_top_gru": f.get("accuracy", 0) >= results.get("top_gru", {}).get("accuracy", 1),
        "no_op_recovers_no_gain": results.get("no_op", {}).get("accuracy", 1) <= f.get("accuracy", 0) + 1e-6,
        "phase_scrambled_underperforms": results.get("phase_scrambled", {}).get("accuracy", 1) <= f.get("accuracy", 0) + 1e-6,
        "sigma_decode_above_chance": f.get("sigma_decode", 0) > 0.5,
        # §14: the falsifier is registered on PHASE DISTANCE (token-age is the secondary axis)
        "recency_gradient_negative": f.get("recency", {}).get("corr_b_dphi", 0) < 0,
        "null_pi_small": results.get("full", {}).get("null_pi", 1.0) < 0.5,
    }
    return {"metrics": results, "checks": checks, "passed": all(checks.values())}
