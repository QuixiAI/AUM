"""Phase 3: silence wired into the live model.

- exclude_current read (§4) matches a strictly-causal brute force (reads S_{t-1}).
- AumLMHeadModel with silence_enabled runs forward + backward end-to-end (CPU and MPS),
  producing logits and the SilenceAux the losses/diagnostics consume.
"""

import pytest
import torch

from aum_ssm.models.config_aum import AumConfig
from aum_ssm.models.aum_lm import AumLMHeadModel
from aum_ssm.modules.ssd_reference import (
    aum_state_readout_ref, aum_dynamics, _rotate_ladder, _l2norm,
)

torch.manual_seed(0)


def _brute_exclusive_read(query, k, v, tau_bar, lam_bar, r, theta, dt_bias, eps=1e-4):
    tau, alog, rho, dphi = aum_dynamics(tau_bar, lam_bar, r, theta, dt_bias, eps)
    phi = torch.cumsum(dphi, dim=1)
    B, L, H, Dqk = query.shape
    Dv = v.shape[-1]
    S = torch.zeros(B, H, Dv, Dqk, dtype=query.dtype)
    outs = []
    for t in range(L):
        q_rot = _rotate_ladder(query[:, t], phi[:, t])
        outs.append(torch.einsum("bhpn,bhn->bhp", S, q_rot))          # read S_{t-1} (before write)
        k_rot = _rotate_ladder(_l2norm(k[:, t]), phi[:, t])
        v_hat = _l2norm(v[:, t])
        w = (rho[:, t] * tau[:, t]).unsqueeze(-1).unsqueeze(-1)
        S = torch.exp(alog[:, t]).unsqueeze(-1).unsqueeze(-1) * S \
            + w * (v_hat.unsqueeze(-1) * k_rot.unsqueeze(-2))
    return torch.stack(outs, dim=1)


def test_segment_checkpointing_is_exact():
    # C7 memory fix: gradient-checkpointed segments must reproduce the non-checkpointed loop
    # EXACTLY — same j* (halt_u is pre-drawn), same loss, same gradients.
    def run(seg):
        torch.manual_seed(0)
        cfg = AumConfig(n_layer=2, vocab_size=128, d_intermediate=128, silence_enabled=True,
                        silence_segment=seg)
        m = AumLMHeadModel(cfg).train()
        torch.manual_seed(1)
        ids = torch.randint(0, 128, (2, 12))
        torch.manual_seed(2)                                  # aligns the halt_u draw
        res, aux = m(ids, return_aux=True)
        loss = (torch.nn.functional.cross_entropy(
                    res.logits[:, :-1].reshape(-1, 128), ids[:, 1:].reshape(-1))
                + aux.pi.mean() + aux.E_traj.mean())
        loss.backward()
        grad = m.backbone.layers[-1].unfold.in_proj_qkv.weight.grad.clone()
        return float(loss.detach()), grad, aux.j_star.clone()

    l_plain, g_plain, j_plain = run(0)                        # one unsegmented pass
    l_ckpt, g_ckpt, j_ckpt = run(4)                           # 3 checkpointed segments (L=12)
    assert torch.equal(j_plain, j_ckpt)                       # identical sampled candidates
    assert abs(l_plain - l_ckpt) < 1e-6
    assert torch.allclose(g_plain, g_ckpt, rtol=1e-5, atol=1e-7)


def test_exclude_current_matches_bruteforce():
    g = torch.Generator().manual_seed(3)
    B, L, H, Dqk, Dv = 2, 8, 3, 8, 6
    rnd = lambda *s: torch.randn(*s, dtype=torch.float64, generator=g)
    query, k, v = rnd(B, L, H, Dqk), rnd(B, L, H, Dqk), rnd(B, L, H, Dv)
    tb, lb, r, th, dtb = rnd(B, L, H), rnd(B, L, H), rnd(B, L, H), rnd(B, L, H), rnd(H)
    ro = aum_state_readout_ref(query, k, v, tb, lb, r, th, dt_bias=dtb, block_len=4,
                               exclude_current=True)
    brute = _brute_exclusive_read(query, k, v, tb, lb, r, th, dtb)
    assert torch.allclose(ro, brute, rtol=1e-9, atol=1e-9), (ro - brute).abs().max().item()


def _tiny(silence):
    return AumConfig(n_layer=2, vocab_size=512, d_intermediate=256, silence_enabled=silence)


def _run_model(device, silence):
    torch.manual_seed(0)
    model = AumLMHeadModel(_tiny(silence)).to(device)
    ids = torch.randint(0, 512, (2, 16), device=device)
    result, aux = model(ids, return_aux=True)
    assert result.logits.shape == (2, 16, 512)
    if silence:
        assert aux is not None and aux.sigma_star.shape == (2, 16, model.config.d_sigma)
        assert torch.isfinite(aux.pi).all()
    else:
        assert aux is None
    loss = torch.nn.functional.cross_entropy(
        result.logits[:, :-1].reshape(-1, 512), ids[:, 1:].reshape(-1))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    return loss.item()


def test_silence_forward_backward_cpu():
    assert _run_model("cpu", silence=True) > 0
    assert _run_model("cpu", silence=False) > 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_silence_forward_backward_mps():
    assert _run_model("mps", silence=True) > 0


if __name__ == "__main__":
    test_exclude_current_matches_bruteforce(); print("exclude_current OK")
    print("cpu silence loss:", _run_model("cpu", True))
    if torch.backends.mps.is_available():
        print("mps silence loss:", _run_model("mps", True))
