"""Metal backend (M2/M3): the AUM U-phase via tk_torch's mamba2 Metal kernel on MPS, validated
against the pure-PyTorch reference. Requires ThunderMittens (tk_torch) + an Apple GPU; skips otherwise."""

import sys

import pytest
import torch

sys.path.insert(0, "/Users/eric/ThunderMittens/ThunderMittens/kernels")
pytest.importorskip("tk_torch")
pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")

from aum_ssm.ops.metal.unfold_metal import unfold_metal_chunk
from aum_ssm.modules.ssd_reference import aum_unfold_chunk_ref

_KEYS = ("q", "k", "v", "tau_bar", "lam_bar", "r", "theta")


def _inputs(B=2, L=16, H=8, D=64, z=True, Dskip=True, grad=False):
    g = torch.Generator().manual_seed(0)
    mk = lambda *s: torch.randn(*s, generator=g)
    d = dict(q=mk(B, L, H, D), k=mk(B, L, H, D), v=mk(B, L, H, D),
             tau_bar=mk(B, L, H), lam_bar=mk(B, L, H), r=mk(B, L, H), theta=mk(B, L, H),
             dt_bias=mk(H))
    d["z"] = mk(B, L, H, D) if z else None
    d["D"] = mk(H, D) if Dskip else None
    d = {k: (v.to("mps").float() if torch.is_tensor(v) else v) for k, v in d.items()}
    if grad:
        for k in _KEYS:
            d[k].requires_grad_(True)
    return d


def test_metal_forward_matches_reference():
    inp = _inputs(z=True, Dskip=True)
    kw = {k: inp[k] for k in _KEYS} | dict(z=inp["z"], D=inp["D"], dt_bias=inp["dt_bias"])
    y_metal = unfold_metal_chunk(**kw)
    y_ref, _ = aum_unfold_chunk_ref(block_len=16, **kw)
    torch.mps.synchronize()
    rel = (y_metal.float() - y_ref.float()).abs().max() / (y_ref.float().abs().max() + 1e-6)
    assert float(rel) < 0.03, float(rel)


def test_metal_grad_matches_reference():
    # no z / no D-skip -> the output is the raw SSD readout, so grads compare fp32-vs-fp32 tightly
    a = _inputs(z=False, Dskip=False, grad=True)
    b = _inputs(z=False, Dskip=False, grad=True)
    unfold_metal_chunk(**{k: a[k] for k in _KEYS}, dt_bias=a["dt_bias"]).float().sum().backward()
    aum_unfold_chunk_ref(block_len=16, **{k: b[k] for k in _KEYS}, dt_bias=b["dt_bias"])[0].sum().backward()
    torch.mps.synchronize()
    for k in ("q", "k", "v", "tau_bar", "theta"):
        rel = (a[k].grad - b[k].grad).abs().max() / (b[k].grad.abs().max() + 1e-6)
        assert float(rel) < 0.02, (k, float(rel))


if __name__ == "__main__":
    test_metal_forward_matches_reference(); print("metal forward OK")
    test_metal_grad_matches_reference(); print("metal grad OK")
