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
