"""Muon + auxiliary-Adam optimizer wiring for AUM-Ø (train/muon.py).

Checks the parameter partition (Muon for 2D hidden matrices; AdamW for the tied embedding/head,
scalars, conv, and vector-shaped heads) covers every trainable parameter exactly once, and that a
real forward/backward/step updates the matrices and stays finite. CPU; MPS Newton-Schulz smoke."""

import pytest
import torch

from aum_ssm.models.config_aum import AumConfig
from aum_ssm.models.aum_lm import AumLMHeadModel
from train.muon import partition_params, build_optimizer, muon_update


def _model():
    torch.manual_seed(0)
    return AumLMHeadModel(AumConfig(n_layer=2, vocab_size=512, d_intermediate=128, silence_enabled=True))


def test_partition_covers_every_param_once():
    m = _model()
    muon, embed, scalar = partition_params(m)
    trainable = {id(p) for p in m.parameters() if p.requires_grad}
    ids = [id(p) for p in muon + embed + scalar]
    assert len(ids) == len(set(ids)) == len(trainable)          # disjoint + complete
    assert muon and embed and scalar
    assert all(p.ndim == 2 and min(p.shape) > 1 for p in muon)  # only true matrices get Muon


def test_partition_routing_by_name():
    m = _model()
    muon, embed, scalar = partition_params(m)
    where = {}
    for tag, ps in (("muon", muon), ("embed", embed), ("scalar", scalar)):
        for p in ps:
            where[id(p)] = tag
    named = dict(m.named_parameters())
    b = lambda n: where[id(named[n])]
    assert b("backbone.embedding.weight") == "embed"
    assert b("backbone.layers.0.mlp.fc1.weight") == "muon"
    assert b("backbone.layers.0.ground_attn.q_proj.weight") == "muon"
    assert b("backbone.layers.0.unfold.out_proj.weight") == "muon"
    assert b("backbone.layers.0.unfold.conv1d.weight") == "scalar"      # ndim 3 (depthwise)
    assert b("backbone.layers.0.unfold.A_log") == "scalar"              # 1D
    assert b("backbone.layers.0.unfold.norm.weight") == "scalar"        # 1D gain


def test_muon_step_updates_matrices_and_stays_finite():
    m = _model().train()
    opt = build_optimizer(m, muon_lr=0.02, embed_lr=1e-3, scalar_lr=1e-3)
    ids = torch.randint(0, 512, (2, 16))
    before = {n: p.detach().clone() for n, p in m.named_parameters()}

    logits = m(ids).logits
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 512), ids[:, 1:].reshape(-1))
    loss.backward()
    opt.step()
    opt.zero_grad()

    assert all(torch.isfinite(p).all() for p in m.parameters())
    gate = "backbone.layers.0.mlp.fc1.weight"                           # a Muon matrix moved
    assert not torch.equal(dict(m.named_parameters())[gate].detach(), before[gate])
    emb = "backbone.embedding.weight"                                   # the Adam-side table moved
    assert not torch.equal(dict(m.named_parameters())[emb].detach(), before[emb])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_newton_schulz_runs_on_mps():
    torch.manual_seed(0)
    g = torch.randn(128, 256, device="mps")
    buf = torch.zeros_like(g)
    upd = muon_update(g.clone(), buf)                                   # bf16 NS iteration on Metal
    torch.mps.synchronize()
    assert upd.shape == g.shape and torch.isfinite(upd).all()
