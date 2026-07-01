"""Phase 2: the AUM-Ø silence block forward (§4-§14), CPU / pure PyTorch.

Driven by a stub evidence_read so the silence logic (predict -> precision -> register ->
J-loop -> consistency -> pressure -> halting -> output) is tested independently of the
U-phase read implementation.
"""

import pytest
import torch

from aum_ssm.modules.silence import SilenceBlock
from aum_ssm.training.losses import consistency_loss

torch.manual_seed(0)


def _stub_read(d_model):
    W = torch.randn(d_model, d_model)
    def read(q, phi=None, exclude_current=False):
        return q @ W
    return read


def _run(d_model=32, d_sigma=8, d_mu=4, d_phase=8, j_max=2, B=2, L=6, H=4,
         ablation=None, with_logits=False):
    blk = SilenceBlock(d_model, d_sigma=d_sigma, d_mu=d_mu, d_phase=d_phase, j_max=j_max)
    g = torch.randn(B, L, d_model, requires_grad=True)
    phi_t = torch.randn(B, L, H)
    phi_prev = torch.randn(B, L, H)
    sigma_prev = torch.randn(B, L, d_sigma)
    m_t = torch.randn(B, L, d_mu)
    s_t = torch.randn(B, L, 1)
    logits_fn = torch.nn.Linear(d_model, 10) if with_logits else None
    o, aux = blk(g, _stub_read(d_model), phi_t, phi_prev, sigma_prev, m_t, s_t,
                 logits_fn=logits_fn, ablation=ablation)
    return blk, g, o, aux


def test_param_count_reference():
    blk = SilenceBlock(512, d_sigma=128, d_mu=32, d_phase=32, j_max=2)
    assert sum(p.numel() for p in blk.parameters()) == 1_769_536


def test_forward_shapes():
    blk, g, o, aux = _run()
    assert o.shape == g.shape
    assert len(aux.sigma_traj) == blk.j_max + 1
    assert aux.E_traj.shape[-1] == blk.j_max + 1
    assert aux.pi.shape == g.shape[:2]
    assert aux.sigma_bar.shape == (*g.shape[:2], blk.d_sigma)


def test_halting_distribution():
    _, _, _, aux = _run()
    w = aux.w
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)   # sum_j w_j = 1
    assert (aux.expected_J >= 0).all() and (aux.expected_J <= 2 + 1e-5).all()


def test_backward_finite():
    blk, g, o, aux = _run(with_logits=True)
    (o.sum() + aux.pi.sum() + aux.E_traj.sum()).backward()
    assert torch.isfinite(g.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in blk.parameters() if p.grad is not None)


def test_no_op_ablation():
    # no-op: revision runs but sigma_bar collapses to sigma^0 (§22 control)
    _, _, _, aux = _run(ablation="no_op")
    assert torch.allclose(aux.sigma_bar, aux.sigma0)


def test_no_read_ablation():
    # no-read: every S-read is exactly zero (§22 control), but sigma still updates from [g,e,mu]
    _, _, _, aux = _run(ablation="no_read")
    assert all(float(r.abs().max()) == 0.0 for r in aux.r_traj)
    assert not torch.allclose(aux.sigma_traj[1], aux.sigma0)  # revision still moved sigma


def test_consistency_loss_nonneg():
    _, _, _, aux = _run()
    loss = consistency_loss(aux.E_traj, lambda_consistency=1.0)
    assert loss.item() >= 0.0


def test_logits_entropy_path():
    # with a classifier, pressure consumes a real predictive entropy H_t
    _, _, _, aux_h = _run(with_logits=True)
    assert torch.isfinite(aux_h.pi).all()


if __name__ == "__main__":
    test_param_count_reference(); print("param count OK")
    for t in (test_forward_shapes, test_halting_distribution, test_backward_finite,
              test_no_op_ablation, test_no_read_ablation, test_consistency_loss_nonneg,
              test_logits_entropy_path):
        t(); print(t.__name__, "OK")
