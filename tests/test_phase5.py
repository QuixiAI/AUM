"""Phase 5: §23 gate harness smoke (CPU). Verifies the machinery — every variant trains,
metrics compute, and the gate table assembles. A real pass/fail needs training to convergence."""

import random

import pytest
import torch

from aum_ssm.training import gate as G


def test_all_variants_train_one_pass():
    # every baseline/ablation/control constructs and takes training steps without error
    for v in G.VARIANTS:
        model, cfg, tr = G.train_variant(v, steps=2, rng=random.Random(0), batch=4, length=10)
        assert model is not None


def test_top_gru_baseline_adds_a_gru():
    ref, _, _ = G.train_variant(G.Variant("full", {"silence_enabled": True}),
                                steps=1, rng=random.Random(0), batch=4, length=10)
    gru, _, _ = G.train_variant(G.Variant("top_gru", {"silence_enabled": True, "baseline": "top_gru"}),
                                steps=1, rng=random.Random(0), batch=4, length=10)
    assert hasattr(gru.backbone.silence, "gru")
    assert sum(p.numel() for p in gru.parameters()) > sum(p.numel() for p in ref.parameters())


def test_evaluate_metrics_finite():
    model, cfg, _ = G.train_variant(G.Variant("full", {"silence_enabled": True}),
                                    steps=3, rng=random.Random(1), batch=8, length=12)
    m = G.evaluate(model, cfg, None, random.Random(2), batch=8)
    assert 0.0 <= m["accuracy"] <= 1.0
    for k in ("mean_pi", "mean_J", "corr_pi_b"):
        assert m[k] == m[k]              # not NaN
    assert -1.0 <= m["corr_pi_b"] <= 1.0


def test_run_gate_assembles_table():
    subset = [
        G.Variant("full", {"silence_enabled": True}),
        G.Variant("evidence_core", {"silence_enabled": False}),
        G.Variant("no_op", {"silence_enabled": True}, "no_op"),
    ]
    out = G.run_gate(steps=3, variants=subset)
    assert set(out) == {"metrics", "checks", "passed"}
    assert "full" in out["metrics"] and isinstance(out["passed"], bool)
    assert all(isinstance(v, bool) for v in out["checks"].values())
    f = out["metrics"]["full"]
    assert "corr_b_dphi" in f["recency"] and "corr_b_age" in f["recency"]   # §14 phase-distance axis
    assert "evidence_survival" in f                                          # §16 survival probe
    assert "corr_pi_b_on_policy" in f                                        # §11 gauge


def test_recency_gradient_reports_phase_distance():
    model, cfg, _ = G.train_variant(G.Variant("full", {"silence_enabled": True}),
                                    steps=2, rng=random.Random(3), batch=4, length=12)
    rec = G.recency_gradient(model, cfg, random.Random(4), ages=(2, 6), batch=4, length=12)
    for k in ("corr_b_dphi", "corr_b_age"):
        assert -1.0 <= rec[k] <= 1.0 or rec[k] != rec[k]


def test_sigma_relevance_and_survival_probe_run():
    from aum_ssm.training import diagnostics as D
    from aum_ssm.training.tasks import synthetic as S
    model, cfg, _ = G.train_variant(G.Variant("full", {"silence_enabled": True}),
                                    steps=2, rng=random.Random(5), batch=4, length=12)
    ids, _ = S.make_batch(S.branch_reversal, random.Random(6), 4, 12, holdout=True)
    rel = D.sigma_relevance(model, ids)                     # §16: g_hat degradation, sigma zeroed
    assert rel == rel and model.training                    # finite; train mode restored
    surv = G.evidence_survival_probe(model, cfg, random.Random(7), batch=8, length=12)
    assert 0.0 <= surv <= 1.0 or surv != surv               # accuracy or nan (degenerate labels)


def test_holdout_generators_use_disjoint_values():
    from aum_ssm.training.tasks import synthetic as S
    rng = random.Random(8)
    tr, _ = S.make_batch(S.branch_reversal, rng, 16, 12, holdout=False)
    ho, _ = S.make_batch(S.branch_reversal, random.Random(8), 16, 12, holdout=True)
    half = S.VAL0 + S.NVAL // 2
    tr_vals = tr[(tr >= S.VAL0) & (tr < S.VAL0 + S.NVAL)]
    ho_vals = ho[(ho >= S.VAL0) & (ho < S.VAL0 + S.NVAL)]
    # evidence values: train draws low half, holdout high half (answers may cross via reversal),
    # so at minimum the value distributions must differ and holdout must use the high half
    assert (ho_vals >= half).float().mean() > 0.5
    assert (tr_vals < half).float().mean() > 0.5


if __name__ == "__main__":
    test_all_variants_train_one_pass(); print("variants OK")
    test_top_gru_baseline_adds_a_gru(); print("top_gru OK")
    test_evaluate_metrics_finite(); print("evaluate OK")
    out = G.run_gate(steps=6)
    print("gate checks:", out["checks"])
