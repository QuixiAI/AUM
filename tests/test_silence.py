"""Phase 2 (v6): the AUM-Ø silence block forward (§5-§9), CPU / pure PyTorch.

Driven by a stub evidence_read so the silence logic (predict -> precision -> register ->
J-loop -> consistency -> pressure -> loss-mixture halting -> output) is tested independently
of the U-phase read implementation. §8: halting mixes LOSSES, never states — per-candidate
outputs + a single carried sigma^{j*}; no sigma blend exists anywhere.
"""

import pytest
import torch

from aum_ssm.modules.silence import SilenceBlock
from aum_ssm.training.losses import consistency_loss

torch.manual_seed(0)


def _stub_read(d_model):
    W = torch.randn(d_model, d_model)

    def read(q, phi=None, exclude_current=False, pooled=False):
        if pooled:                       # query-free Pool(S) stand-in for the Top-GRU baseline
            return W.sum(0).expand(2, 6, d_model)
        return q @ W
    return read


def _run(d_model=32, d_sigma=8, d_mu=4, d_phase=8, j_max=2, B=2, L=6, H=4,
         ablation=None, forced_depth=None, entropy_feature=False, with_logits=False,
         top_gru=False, train=True):
    blk = SilenceBlock(d_model, d_sigma=d_sigma, d_mu=d_mu, d_phase=d_phase, j_max=j_max,
                       entropy_feature=entropy_feature, top_gru=top_gru)
    blk.train(train)
    g = torch.randn(B, L, d_model, requires_grad=True)
    phi_t = torch.randn(B, L, H)
    phi_prev = torch.randn(B, L, H)
    sigma_prev = torch.randn(B, L, d_sigma)
    m_t = torch.randn(B, L, d_mu)
    s_t = torch.randn(B, L, 1)
    logits_fn = torch.nn.Linear(d_model, 10) if with_logits else None
    o, aux = blk(g, _stub_read(d_model), phi_t, phi_prev, sigma_prev, m_t, s_t,
                 logits_fn=logits_fn, ablation=ablation, forced_depth=forced_depth)
    return blk, g, o, aux


def test_param_count_reference():
    # v6: pressure_in is [128, 515] (H_t dropped from the base features, §9) -> -128 vs v5.3
    blk = SilenceBlock(512, d_sigma=128, d_mu=32, d_phase=32, j_max=2)
    assert sum(p.numel() for p in blk.parameters()) == 1_769_408
    # the registered entropy-feature ablation restores the extra column
    blk_h = SilenceBlock(512, d_sigma=128, d_mu=32, d_phase=32, j_max=2, entropy_feature=True)
    assert sum(p.numel() for p in blk_h.parameters()) == 1_769_536


def test_forward_shapes():
    blk, g, o, aux = _run()
    B, L, d = g.shape
    assert o.shape == g.shape
    assert len(aux.sigma_traj) == blk.j_max + 1
    assert aux.E_traj.shape[-1] == blk.j_max + 1
    assert aux.pi.shape == (B, L)
    assert aux.o_stack.shape == (B, L, blk.j_max + 1, d)      # per-candidate outputs (§8)
    assert aux.j_star.shape == (B, L) and aux.j_star.dtype == torch.long
    assert aux.sigma_star.shape == (B, L, blk.d_sigma)


def test_halting_distribution():
    _, _, _, aux = _run()
    w = aux.w
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)   # sum_j w_j = 1
    assert (aux.expected_J >= 0).all() and (aux.expected_J <= 2 + 1e-5).all()


def test_carry_is_single_candidate_not_blend():
    # §8: sigma_star is EXACTLY one sigma^j (per token) and o_t is that candidate's output.
    blk, g, o, aux = _run()
    stack = torch.stack(aux.sigma_traj, dim=-2)               # (B,L,J+1,d_sigma)
    idx = aux.j_star[..., None, None]
    picked = stack.gather(-2, idx.expand(*aux.j_star.shape, 1, blk.d_sigma)).squeeze(-2)
    assert torch.equal(aux.sigma_star, picked)
    picked_o = aux.o_stack.gather(-2, idx.expand(*aux.j_star.shape, 1, g.shape[-1])).squeeze(-2)
    assert torch.equal(o, picked_o)


def test_forced_depth_one_hot():
    # §11/§12: forced_depth=K makes the halting distribution one-hot at K (the fixed-K label path)
    blk, _, _, aux = _run(forced_depth=2)
    assert (aux.j_star == 2).all()
    assert torch.allclose(aux.w[..., 2], torch.ones_like(aux.w[..., 2]))
    assert torch.allclose(aux.expected_J, torch.full_like(aux.expected_J, 2.0))


def test_eval_selection_is_thresholded_and_deterministic():
    torch.manual_seed(1)
    _, _, o1, aux1 = _run(train=False)
    torch.manual_seed(1)
    _, _, o2, aux2 = _run(train=False)
    assert torch.equal(aux1.j_star, aux2.j_star)              # min{j: p_j >= delta}, no sampling
    assert torch.equal(o1, o2)


def test_backward_finite():
    blk, g, o, aux = _run()
    # train through the §10 mixture (w-weighted candidate outputs), as the LM loss does
    mix = (aux.w.unsqueeze(-1) * aux.o_stack).sum()
    (mix + aux.pi.sum() + aux.E_traj.sum()).backward()
    assert torch.isfinite(g.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in blk.parameters() if p.grad is not None)


def test_consistency_detaches_precision():
    # §7: mu enters the consistency functional DETACHED — E's precision weighting sends no
    # gradient to mu (gradient cannot shrink E by turning precision off), while sigma still gets one.
    blk = SilenceBlock(32, d_sigma=8, d_mu=4, d_phase=8, j_max=2)
    g = torch.randn(2, 6, 32)
    sigma = torch.randn(2, 6, 8, requires_grad=True)
    sigma_prev = torch.randn(2, 6, 8)
    r_sigma = torch.randn(2, 6, 32)
    mu = torch.randn(2, 6, 4, requires_grad=True)
    blk._consistency(g, sigma, r_sigma, mu, sigma_prev).sum().backward()
    assert mu.grad is None                                    # the §7 stopgrad
    assert sigma.grad is not None and torch.isfinite(sigma.grad).all()


def test_no_op_ablation():
    # no-op: the loop runs but revision is discarded — every candidate output is sigma^0's (§14)
    _, _, o, aux = _run(ablation="no_op")
    assert torch.allclose(aux.sigma_star, aux.sigma0)


def test_no_read_ablation():
    # no-read: every SILENT S-read is exactly zero (§14), but sigma still updates from [g,e,mu]
    _, _, _, aux = _run(ablation="no_read")
    assert all(float(r.abs().max()) == 0.0 for r in aux.r_traj)
    assert not torch.allclose(aux.sigma_traj[1], aux.sigma0)  # revision still moved sigma


def test_consistency_loss_nonneg():
    _, _, _, aux = _run()
    loss = consistency_loss(aux.E_traj, lambda_consistency=1.0)
    assert loss.item() >= 0.0


def test_entropy_feature_flag():
    # default: no H_t (515-wide pressure input, no logits needed); flag on: H_t consumed (§9/§14)
    _, _, _, aux = _run(with_logits=False)
    assert torch.isfinite(aux.pi).all()
    _, _, _, aux_h = _run(entropy_feature=True, with_logits=True)
    assert torch.isfinite(aux_h.pi).all()


def test_top_gru_pooled_prediction_head():
    # §14: the Top-GRU baseline predicts from Pool(S_{t-1}) — no sigma-conditioned read anywhere,
    # but g_hat is real (not handicapped) so the ablated factor is precisely the silent read.
    blk, g, o, aux = _run(top_gru=True)
    assert torch.isfinite(aux.g_hat).all()
    assert all(float(r.abs().max()) == 0.0 for r in aux.r_traj)   # no S read in the loop
    assert o.shape == g.shape


if __name__ == "__main__":
    for t in (test_param_count_reference, test_forward_shapes, test_halting_distribution,
              test_carry_is_single_candidate_not_blend, test_forced_depth_one_hot,
              test_eval_selection_is_thresholded_and_deterministic, test_backward_finite,
              test_consistency_detaches_precision, test_no_op_ablation, test_no_read_ablation,
              test_consistency_loss_nonneg, test_entropy_feature_flag,
              test_top_gru_pooled_prediction_head):
        t(); print(t.__name__, "OK")
