"""Phase 4: training harness — synthetic tasks, loss wiring, counterfactual benefit, staged
trainer with the pressure gate, and diagnostics. CPU / pure PyTorch."""

import random

import pytest
import torch

from aum_ssm.models.config_aum import AumConfig
from aum_ssm.models.aum_lm import AumLMHeadModel
from aum_ssm.training.tasks import synthetic as S
from aum_ssm.training import losses as L
from aum_ssm.training.counterfactual import rollout_benefit, on_policy_benefit
from aum_ssm.training.schedule import Stage, ScheduleConfig
from aum_ssm.training.trainer import AumTrainer
from aum_ssm.training import diagnostics as D


def _model(vocab=S.VOCAB_SIZE, n_layer=2, silence=True):
    return AumLMHeadModel(AumConfig(vocab_size=vocab, n_layer=n_layer, d_intermediate=128,
                                    silence_enabled=silence))


def test_tasks_shapes_and_range():
    rng = random.Random(0)
    for fn in S.TASKS.values():
        ids, metas = S.make_batch(fn, rng, 4, 12)
        assert ids.shape == (4, 12)
        assert 0 <= int(ids.min()) and int(ids.max()) < S.VOCAB_SIZE
        assert len(metas) == 4


def test_branch_reversal_structure():
    rng = random.Random(1)
    ids, meta = S.branch_reversal(rng, 12, event_distance=2)
    assert meta["event_pos"] == 10 and meta["answer_pos"] == 11
    assert S.VAL0 <= int(ids[meta["answer_pos"]]) < S.VAL0 + S.NVAL


def test_total_loss_sums_active_only():
    a, b = torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)
    total, used = L.total_loss({"lm": a, "pred": b, "compute": torch.tensor(9.0)}, {"lm", "pred"})
    assert float(total.detach()) == 3.0 and set(used) == {"lm", "pred"}


def test_rollout_benefit_deterministic():
    torch.manual_seed(0)
    m = _model()
    rng = random.Random(2)
    ids, _ = S.make_batch(S.branch_reversal, rng, 3, 12)
    b1, y1, aux, res = rollout_benefit(m, ids, beta=0.02)
    b2, y2, _, _ = rollout_benefit(m, ids, beta=0.02)
    assert torch.allclose(b1, b2)                 # paired: deterministic on repeat
    assert (y1 >= 0).all() and b1.shape == (3, 11)
    assert m.backbone.silence_enabled is True     # restored after the frozen-off branch


def test_on_policy_gauge_smoke():
    # §11: the residual policy-gap is measured, not ignored — the gauge runs the LIVE policy.
    torch.manual_seed(0)
    m = _model()
    rng = random.Random(9)
    ids, _ = S.make_batch(S.branch_reversal, rng, 3, 12)
    b_pol, aux = on_policy_benefit(m, ids)
    assert b_pol.shape == (3, 11) and torch.isfinite(b_pol).all()
    assert m.training is True                              # mode restored


def test_pressure_gate_blocks_until_pred_good():
    # v6 §12: the gate is a scale-free R^2 margin — clears only when pred R^2 > eta_r2.
    m = _model()
    tr = AumTrainer(m, torch.optim.Adam(m.parameters(), 1e-3), m.config,
                    ScheduleConfig(eta_r2=0.15))
    assert tr.maybe_advance_stage(0.05) == Stage.EVIDENCE_CORE   # below the margin: blocked
    assert tr.maybe_advance_stage(0.40) == Stage.FORCED_REVISION # above: clears


def test_pred_val_r2_is_scale_free_metric():
    torch.manual_seed(0)
    m = _model()
    tr = AumTrainer(m, torch.optim.Adam(m.parameters(), 1e-3), m.config, ScheduleConfig())
    rng = random.Random(7)
    ids, _ = S.make_batch(S.branch_reversal, rng, 4, 12)
    r2 = tr.pred_val_r2(ids)
    assert r2 <= 1.0 and torch.isfinite(torch.tensor(r2))        # untrained: typically << eta_r2


def test_train_step_stage1_then_stage2():
    torch.manual_seed(0)
    m = _model()
    tr = AumTrainer(m, torch.optim.Adam(m.parameters(), 1e-3), m.config,
                    ScheduleConfig(eta_r2=-1e9))                 # gate always clears (smoke)
    rng = random.Random(3)
    ids, _ = S.make_batch(S.branch_reversal, rng, 4, 12)
    met1, _ = tr.train_step(ids)
    assert "lm" in met1["parts"] and "pred" in met1["parts"] and "pressure" not in met1["parts"]
    tr.maybe_advance_stage(1.0)                                  # -> FORCED_REVISION
    met2, aux2 = tr.train_step(ids)
    assert "pressure" in met2["parts"] and "benefit_mean" in met2
    assert (aux2.j_star == tr.schedule.forced_K).all()           # stage 2 trains AT depth K (§12)


def test_lm_mixture_loss_matches_plain_ce_when_one_hot():
    # w one-hot at j -> the mixture reduces to plain CE on that candidate's logits (§8/§10)
    torch.manual_seed(0)
    B, T, J1, d, V = 2, 5, 3, 16, 11
    o = torch.randn(B, T, J1, d)
    head = torch.nn.Linear(d, V, bias=False)
    targets = torch.randint(0, V, (B, T))
    w = torch.nn.functional.one_hot(torch.full((B, T), 2), J1).float()
    mix = L.lm_mixture_loss(o, w, head, targets)
    plain = L.lm_loss(head(o[:, :, 2]), targets)
    assert torch.allclose(mix, plain, atol=1e-6)


def test_precision_loss_is_per_layer_only():
    mus = [torch.rand(2, 6, 4), torch.rand(2, 6, 4)]
    val = L.precision_loss(mus, 1e-3)
    assert float(val) > 0
    # trainer collects layer stashes, never aux.mu (the global field) — assert the wiring exists
    m = _model()
    _ = m(torch.randint(0, S.VOCAB_SIZE, (2, 8)))
    assert all(l._last_mu is not None for l in m.backbone.layers)


def test_overfit_smoke():
    torch.manual_seed(0)
    m = _model()
    tr = AumTrainer(m, torch.optim.Adam(m.parameters(), 1e-3), m.config, ScheduleConfig())
    rng = random.Random(4)
    ids, _ = S.make_batch(S.branch_reversal, rng, 4, 12)
    first = tr.train_step(ids)[0]["loss"]
    last = first
    for _ in range(60):
        last = tr.train_step(ids)[0]["loss"]
    assert last < first * 0.6, (first, last)


def test_diagnostics():
    pi, b = torch.rand(3, 12), torch.rand(3, 11)
    assert -1.0 <= D.corr_pi_benefit(pi, b) <= 1.0
    sigma = torch.randn(40, 8)
    labels = (sigma[:, 0] > 0).long()               # linearly decodable
    acc = D.sigma_decode_probe(sigma, labels, n_classes=2, epochs=150)
    assert 0.0 <= acc <= 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(name, "OK")
