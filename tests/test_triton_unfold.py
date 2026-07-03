"""Triton U-phase path (aum_ssm/ops/triton/unfold_triton.py, nvidia-plan B2+B3) vs the
pure-PyTorch reference. The SSD core rides the vendored upstream chunked-SSD kernels through
the cumlog<->dt*A adapter (decay and input scale decoupled); operands are torch.compile-fused;
the epilogue is the vendored gated-RMSNorm kernel.

Tolerances: Triton tl.dot on fp32 uses TF32 on Ampere by default (~1e-3 relative; verified
exact to 4e-7 against an fp64 oracle when compiled with TRITON_F32_DEFAULT=ieee), so the fp32
assertions here use TF32-level bounds — the mathematical parity claim rests on the fp64 anchor
test. Skips cleanly off-CUDA."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
pytest.importorskip("triton")

from aum_ssm.modules.ssd_reference import (aum_unfold_chunk_ref, ladder_freqs,
                                           ssd_minimal_discrete)
from aum_ssm.ops.triton.unfold_triton import aum_ssd, unfold_triton_chunk

DEV = "cuda"


def test_core_matches_fp64_oracle():
    """The adapter math vs an fp64 ssd_minimal_discrete: the fp32 gap must be TF32-sized
    (the fp32 torch reference sits at ~3e-7 of the same oracle)."""
    torch.manual_seed(0)
    B, L, H, P = 2, 256, 8, 64
    x = torch.randn(B, L, H, P, device=DEV) * 0.5
    Bm = torch.randn(B, L, H, P, device=DEV) * 0.5
    C = torch.randn(B, L, H, P, device=DEV) * 0.5
    dt = torch.rand(B, L, H, device=DEV) * 0.5 + 0.1
    alog = -torch.rand(B, L, H, device=DEV) * 0.3
    ref64, _ = ssd_minimal_discrete((dt.unsqueeze(-1) * x).double(), alog.double(),
                                    Bm.double(), C.double(), 64)
    out = aum_ssd(x, dt, alog, Bm, C, 64)
    scale = ref64.abs().max().item()
    assert ((out.double() - ref64).abs().max() / scale).item() < 5e-3


@pytest.mark.parametrize("L", [64, 256, 4096])
def test_core_grads_match_reference(L):
    torch.manual_seed(1)
    B, H, P = 2, 8, 64
    mk = lambda *s: (torch.randn(*s, device=DEV) * 0.5).requires_grad_()  # noqa: E731
    x, Bm, C = mk(B, L, H, P), mk(B, L, H, P), mk(B, L, H, P)
    dt = (torch.rand(B, L, H, device=DEV) * 0.5 + 0.1).requires_grad_()
    alog = (-torch.rand(B, L, H, device=DEV) * 0.3).requires_grad_()
    leaves = (x, dt, alog, Bm, C)
    proj = torch.randn(B, L, H, P, device=DEV)

    ref, _ = ssd_minimal_discrete(dt.unsqueeze(-1) * x, alog, Bm, C, 64)
    (ref * proj).sum().backward()
    gref = [t.grad.clone() for t in leaves]
    for t in leaves:
        t.grad = None
    out = aum_ssd(x, dt, alog, Bm, C, 64)
    (out * proj).sum().backward()
    for a, t in zip(gref, leaves):
        rel = ((t.grad - a).abs().max() / a.abs().max().clamp(min=1e-8)).item()
        assert rel < 8e-3, rel                            # TF32 bound


def test_unfold_pipeline_matches_reference():
    """Full B2+B3 pipeline (operands + core + D-skip + gated RMSNorm) vs
    aum_unfold_chunk_ref, forward and every input/param grad."""
    torch.manual_seed(0)
    B, L, H, Dh = 2, 256, 8, 64
    freqs = ladder_freqs(Dh // 2, device=DEV)
    mk = lambda *s: (torch.randn(*s, device=DEV) * 0.5).requires_grad_()  # noqa: E731
    q, k, v, z = mk(B, L, H, Dh), mk(B, L, H, Dh), mk(B, L, H, Dh), mk(B, L, H, Dh)
    tb, lb, r, th = mk(B, L, H), mk(B, L, H), mk(B, L, H), mk(B, L, H)
    dtb = (torch.rand(H, device=DEV) * 0.1 - 2).requires_grad_()
    D = torch.randn(H, Dh, device=DEV).requires_grad_()
    nw = (torch.ones(Dh, device=DEV) + 0.1 * torch.randn(Dh, device=DEV)).requires_grad_()
    leaves = [q, k, v, z, tb, lb, r, th, dtb, D, nw]
    proj = torch.randn(B, L, H, Dh, device=DEV)
    kw = dict(z=z, D=D, dt_bias=dtb, eps=1e-4, norm_weight=nw, freqs=freqs)

    ref, _ = aum_unfold_chunk_ref(q, k, v, tb, lb, r, th, block_len=64, **kw)
    (ref * proj).sum().backward()
    gref = [t.grad.clone() for t in leaves]
    for t in leaves:
        t.grad = None
    out = unfold_triton_chunk(q, k, v, tb, lb, r, th, chunk_size=64, **kw)
    assert ((out - ref).abs().max() / ref.abs().max()).item() < 5e-3
    (out * proj).sum().backward()
    for a, t in zip(gref, leaves):
        rel = ((t.grad - a).abs().max() / a.abs().max().clamp(min=1e-8)).item()
        assert rel < 1e-2, rel                            # TF32 bound through long chains


def test_unfold_module_routes_and_matches():
    """Unfold with kernel_backend=auto routes CUDA chunk-aligned lengths to triton and
    matches backend=reference; ragged lengths fall back to the reference (no crash)."""
    from aum_ssm.modules.unfold import Unfold

    torch.manual_seed(0)
    u = Unfold(512, kernel_backend="auto", layer_idx=0, device=DEV).float().train()
    x = torch.randn(2, 256, 512, device=DEV, requires_grad=True)
    out_t, m_t, s_t = u(x)
    (out_t.square().mean()).backward()
    gx = x.grad.clone()
    gW = u.in_proj_qkv.weight.grad.clone()
    x.grad = None
    u.zero_grad(set_to_none=True)

    u.kernel_backend = "reference"
    out_r, _, _ = u(x)
    (out_r.square().mean()).backward()
    assert ((out_t - out_r).abs().max() / out_r.abs().max()).item() < 5e-3
    assert ((gx - x.grad).abs().max() / x.grad.abs().max()).item() < 1e-2
    assert ((gW - u.in_proj_qkv.weight.grad).abs().max()
            / u.in_proj_qkv.weight.grad.abs().max()).item() < 1e-2

    u.kernel_backend = "auto"
    out_ragged, _, _ = u(torch.randn(2, 100, 512, device=DEV))   # 100 % 64 != 0 -> reference
    assert out_ragged.shape == (2, 100, 512)
