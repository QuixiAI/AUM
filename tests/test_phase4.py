"""Phase 4: training harness — synthetic tasks, loss wiring, counterfactual benefit, staged
trainer with the pressure gate, and diagnostics. CPU / pure PyTorch."""

import random

import pytest
import torch

from aum_ssm.models.config_aum import AumConfig
from aum_ssm.models.aum_lm import AumLMHeadModel
from aum_ssm.training.tasks import synthetic as S
from aum_ssm.training import losses as L
from aum_ssm.training.counterfactual import rollout_benefit
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


def test_pressure_gate_blocks_until_pred_good():
    m = _model()
    tr = AumTrainer(m, torch.optim.Adam(m.parameters(), 1e-3), m.config,
                    ScheduleConfig(pred_val_gate_eta=0.0))       # never clears
    assert tr.maybe_advance_stage(5.0) == Stage.EVIDENCE_CORE
    tr.schedule.pred_val_gate_eta = 100.0                        # now clears
    assert tr.maybe_advance_stage(5.0) == Stage.FORCED_REVISION


def test_train_step_stage1_then_stage2():
    torch.manual_seed(0)
    m = _model()
    tr = AumTrainer(m, torch.optim.Adam(m.parameters(), 1e-3), m.config,
                    ScheduleConfig(pred_val_gate_eta=100.0))
    rng = random.Random(3)
    ids, _ = S.make_batch(S.branch_reversal, rng, 4, 12)
    met1, _ = tr.train_step(ids)
    assert "lm" in met1["parts"] and "pred" in met1["parts"] and "pressure" not in met1["parts"]
    tr.maybe_advance_stage(1.0)                                  # -> FORCED_REVISION
    met2, _ = tr.train_step(ids)
    assert "pressure" in met2["parts"] and "benefit_mean" in met2


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
