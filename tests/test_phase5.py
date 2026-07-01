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


if __name__ == "__main__":
    test_all_variants_train_one_pass(); print("variants OK")
    test_top_gru_baseline_adds_a_gru(); print("top_gru OK")
    test_evaluate_metrics_finite(); print("evaluate OK")
    out = G.run_gate(steps=6)
    print("gate checks:", out["checks"])
