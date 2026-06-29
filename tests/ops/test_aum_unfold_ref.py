"""CPU equivalence tests for the AUM-Ø U-phase reference (§6).

These run on a Mac with CPU torch and NO Triton: they validate that the serial
(step) reference and the chunk-parallel reference compute the same recurrence,
and that the swapped-query state readout (the silence read, §10) is correct.
"""

import math

import pytest
import torch

from aum_ssm.modules.ssd_reference import (
    aum_unfold_step_ref,
    aum_unfold_chunk_ref,
    aum_state_readout_ref,
    aum_dynamics,
    _rotate_single_phase,
    _l2norm,
)

torch.manual_seed(0)


def _rand_inputs(B=2, L=8, H=3, Dqk=8, Dv=6, dtype=torch.float64):
    g = torch.Generator().manual_seed(1234)
    r = lambda *s: torch.randn(*s, dtype=dtype, generator=g)
    return dict(
        q=r(B, L, H, Dqk), k=r(B, L, H, Dqk), v=r(B, L, H, Dv), z=r(B, L, H, Dv),
        tau_bar=r(B, L, H), lam_bar=r(B, L, H), r=r(B, L, H), theta=r(B, L, H),
        D=r(H, Dv), dt_bias=r(H),
    )


def _brute_readout(query, inp, eps=1e-4):
    """Independent serial computation of r_t = S_t R(phi_t) query_t (oracle for the readout)."""
    B, L, H, Dqk = query.shape
    Dv = inp["v"].shape[-1]
    S = torch.zeros(B, H, Dv, Dqk, dtype=query.dtype)
    phi = torch.zeros(B, H, dtype=query.dtype)
    outs = []
    for t in range(L):
        tau, alog, rho, dphi = aum_dynamics(
            inp["tau_bar"][:, t], inp["lam_bar"][:, t], inp["r"][:, t], inp["theta"][:, t],
            inp["dt_bias"], eps,
        )
        phi = phi + dphi
        k_rot = _rotate_single_phase(_l2norm(inp["k"][:, t]), phi)
        v_hat = _l2norm(inp["v"][:, t])
        w = (rho * tau).unsqueeze(-1).unsqueeze(-1)
        S = torch.exp(alog).unsqueeze(-1).unsqueeze(-1) * S
        S = S + w * (v_hat.unsqueeze(-1) * k_rot.unsqueeze(-2))
        q_rot = _rotate_single_phase(query[:, t], phi)
        outs.append(torch.einsum("bhpn,bhn->bhp", S, q_rot))
    return torch.stack(outs, dim=1)


@pytest.mark.parametrize("block_len", [2, 4, 8])
def test_step_equals_chunk(block_len):
    inp = _rand_inputs()
    h_step, (S_step, phi_step) = aum_unfold_step_ref(**inp)
    h_chunk, (S_chunk, phi_chunk) = aum_unfold_chunk_ref(block_len=block_len, **inp)
    assert torch.allclose(h_step, h_chunk, rtol=1e-9, atol=1e-9), \
        (h_step - h_chunk).abs().max().item()
    assert torch.allclose(S_step, S_chunk, rtol=1e-9, atol=1e-9)
    assert torch.allclose(phi_step, phi_chunk, rtol=1e-9, atol=1e-9)


def test_step_equals_chunk_no_D_no_z():
    inp = _rand_inputs()
    inp["D"] = None
    inp["z"] = None
    h_step, _ = aum_unfold_step_ref(**inp)
    h_chunk, _ = aum_unfold_chunk_ref(block_len=4, **inp)
    assert torch.allclose(h_step, h_chunk, rtol=1e-9, atol=1e-9)


def test_state_readout_matches_brute_force():
    inp = _rand_inputs()
    g = torch.Generator().manual_seed(99)
    query = torch.randn(2, 8, 3, 8, dtype=torch.float64, generator=g)
    ro = aum_state_readout_ref(
        query, inp["k"], inp["v"], inp["tau_bar"], inp["lam_bar"], inp["r"], inp["theta"],
        dt_bias=inp["dt_bias"], block_len=4,
    )
    brute = _brute_readout(query, inp)
    assert torch.allclose(ro, brute, rtol=1e-9, atol=1e-9), (ro - brute).abs().max().item()


def test_readout_with_own_query_equals_forward_readout():
    # With query=q and no D/z, the readout equals the U-phase readout before D-skip/norm.
    inp = _rand_inputs()
    inp_noDz = {**inp, "D": None, "z": None}
    h_chunk, _ = aum_unfold_chunk_ref(block_len=4, **inp_noDz)
    ro = aum_state_readout_ref(
        inp["q"], inp["k"], inp["v"], inp["tau_bar"], inp["lam_bar"], inp["r"], inp["theta"],
        dt_bias=inp["dt_bias"], block_len=4,
    )
    assert torch.allclose(h_chunk, ro, rtol=1e-9, atol=1e-9)


def test_rotation_is_orthogonal():
    # R(phi) preserves norm: ||R(phi) x|| == ||x|| (the rotation-invariance the kernel plan relies on).
    g = torch.Generator().manual_seed(7)
    x = torch.randn(4, 8, dtype=torch.float64, generator=g)
    phi = torch.randn(4, dtype=torch.float64, generator=g)
    xr = _rotate_single_phase(x, phi)
    assert torch.allclose(x.norm(dim=-1), xr.norm(dim=-1), rtol=1e-12, atol=1e-12)


def test_backward_is_finite():
    inp = _rand_inputs(dtype=torch.float64)
    for key in ("q", "k", "v", "tau_bar", "lam_bar", "r", "theta"):
        inp[key] = inp[key].clone().requires_grad_(True)
    h, _ = aum_unfold_chunk_ref(block_len=4, **inp)
    h.sum().backward()
    for key in ("q", "k", "v", "tau_bar", "lam_bar", "r", "theta"):
        assert inp[key].grad is not None and torch.isfinite(inp[key].grad).all()


if __name__ == "__main__":
    for bl in (2, 4, 8):
        test_step_equals_chunk(bl)
    test_step_equals_chunk_no_D_no_z()
    test_state_readout_matches_brute_force()
    test_readout_with_own_query_equals_forward_readout()
    test_rotation_is_orthogonal()
    test_backward_is_finite()
    print("all AUM unfold reference tests passed")
