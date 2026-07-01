"""Metal backend (M2/M3): the AUM U-phase via tk_torch's mamba2 Metal kernel on MPS, validated
against the pure-PyTorch reference — at both headdim 64 and the Appendix-A headdim 128, and
end-to-end in the full model. Requires ThunderMittens (tk_torch) + an Apple GPU; skips otherwise."""

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


@pytest.mark.parametrize("H,D", [(8, 64), (4, 128)])   # both give d_inner = 512
def test_metal_forward_matches_reference(H, D):
    inp = _inputs(H=H, D=D, z=True, Dskip=True)
    kw = {k: inp[k] for k in _KEYS} | dict(z=inp["z"], D=inp["D"], dt_bias=inp["dt_bias"])
    y_metal = unfold_metal_chunk(**kw)
    y_ref, _ = aum_unfold_chunk_ref(block_len=16, **kw)
    torch.mps.synchronize()
    rel = (y_metal.float() - y_ref.float()).abs().max() / (y_ref.float().abs().max() + 1e-6)
    assert float(rel) < 0.03, float(rel)


@pytest.mark.parametrize("H,D", [(8, 64), (4, 128)])
def test_metal_grad_matches_reference(H, D):
    # backward runs the fused bf16 mamba2_bwd kernel; grads match the fp32 reference within bf16
    # tolerance (tau_bar amplifies the kernel's ~1% error through the cumlog/rotary chain).
    a = _inputs(H=H, D=D, z=False, Dskip=False, grad=True)
    b = _inputs(H=H, D=D, z=False, Dskip=False, grad=True)
    unfold_metal_chunk(**{k: a[k] for k in _KEYS}, dt_bias=a["dt_bias"]).float().sum().backward()
    aum_unfold_chunk_ref(block_len=16, **{k: b[k] for k in _KEYS}, dt_bias=b["dt_bias"])[0].sum().backward()
    torch.mps.synchronize()
    for k in ("q", "k", "v", "tau_bar", "theta"):
        rel = (a[k].grad - b[k].grad).abs().max() / (b[k].grad.abs().max() + 1e-6)
        assert float(rel) < 0.06, (k, float(rel))


@pytest.mark.parametrize("D", [64, 128])
def test_mamba2_bwd_kernel_matches_autograd(D):
    # the fused SSD backward kernel directly vs autograd-through the PyTorch SSD core (tight)
    import tk_torch

    def fwd(C, B, X, cl):
        N = C.shape[-2]
        s = C.float() @ B.float().transpose(-1, -2)
        d = torch.exp(cl[..., :, None] - cl[..., None, :])
        m = torch.tril(torch.ones(N, N, device=C.device))
        return (s * d * m) @ X.float()

    torch.manual_seed(0)
    Bt, H, N = 2, 4, 16
    C = torch.randn(Bt, H, N, D, device="mps") * 0.5
    Bm = torch.randn(Bt, H, N, D, device="mps") * 0.5
    X = torch.randn(Bt, H, N, D, device="mps")
    a = torch.sigmoid(torch.randn(Bt, H, N, device="mps")) * 0.5 + 0.5
    cl = torch.cumsum(torch.log(a), -1).float()
    dY = torch.randn(Bt, H, N, D, device="mps")
    Cr, Br, Xr, clr = (t.clone().detach().requires_grad_() for t in (C, Bm, X, cl))
    fwd(Cr, Br, Xr, clr).backward(dY.float())
    dC, dB, dX = tk_torch.mamba2_bwd(C.bfloat16(), Bm.bfloat16(), X.bfloat16(), cl, dY.bfloat16())
    Y = tk_torch.mamba2(C.bfloat16(), Bm.bfloat16(), X.bfloat16(), cl).float()
    torch.mps.synchronize()
    dcl = (dY.float() * Y).sum(-1) - (dX.float() * X.float()).sum(-1)   # host identity
    rel = lambda u, v: ((u - v).abs().max() / (v.abs().max() + 1e-6)).item()
    assert rel(dC.float(), Cr.grad) < 0.02, ("dC", rel(dC.float(), Cr.grad))
    assert rel(dB.float(), Br.grad) < 0.02, ("dB", rel(dB.float(), Br.grad))
    assert rel(dX.float(), Xr.grad) < 0.02, ("dX", rel(dX.float(), Xr.grad))
    assert rel(dcl, clr.grad) < 0.02, ("dcumlog", rel(dcl, clr.grad))


def test_full_model_on_metal_backend():
    # the Appendix-A reference config (4 heads x 128) running on the metal kernel end-to-end
    from aum_ssm.models.config_aum import AumConfig
    from aum_ssm.models.aum_lm import AumLMHeadModel
    torch.manual_seed(0)
    model = AumLMHeadModel(AumConfig(n_layer=2, vocab_size=512, d_intermediate=128)).to("mps")
    ids = torch.randint(0, 512, (2, 16), device="mps")

    for layer in model.backbone.layers:
        layer.unfold.kernel_backend = "reference"
    y_ref = model(ids).logits.detach()
    for layer in model.backbone.layers:
        layer.unfold.kernel_backend = "metal"
    y_metal = model(ids).logits
    torch.mps.synchronize()
    rel = (y_metal.detach().float() - y_ref.float()).abs().max() / (y_ref.abs().max() + 1e-6)
    assert float(rel) < 0.06, float(rel)

    loss = torch.nn.functional.cross_entropy(y_metal[:, :-1].reshape(-1, 512), ids[:, 1:].reshape(-1))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


if __name__ == "__main__":
    for H, D in [(8, 64), (4, 128)]:
        test_metal_forward_matches_reference(H, D)
        test_metal_grad_matches_reference(H, D)
    test_full_model_on_metal_backend()
    print("metal backend (D=64 + D=128 + full model) OK")
