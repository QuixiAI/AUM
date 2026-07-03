"""Triton fused cross-entropy (kernels/triton/cross_entropy.py, roadmap B4) vs torch — raw
kernel parity (incl. ignore_index / label smoothing / z-loss / softcap) and the §8 mixture-CE
fused-linear path vs the dense fallback. Skips cleanly off-CUDA."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
kt = pytest.importorskip("kernels.triton")

DEV = "cuda"


@pytest.mark.parametrize("T,V", [(64, 501), (128, 49152)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ce_kernel_matches_torch(T, V, dtype):
    torch.manual_seed(0)
    logits = torch.randn(T, V, device=DEV, dtype=dtype) * 3
    tgt = torch.randint(0, V, (T,), device=DEV, dtype=torch.int32)
    tgt[::7] = -100
    loss, lse = kt.cross_entropy_fwd(logits, tgt)
    ref = F.cross_entropy(logits.float(), tgt.long(), reduction="none", ignore_index=-100)
    assert (loss - ref).abs().max().item() < 5e-5 * max(1.0, ref.abs().max().item())
    go = torch.rand(T, device=DEV)
    g = kt.cross_entropy_bwd(logits, tgt, lse, go)
    lf = logits.float().detach().requires_grad_()
    (F.cross_entropy(lf, tgt.long(), reduction="none", ignore_index=-100) * go).sum().backward()
    tol = 5e-3 if dtype == torch.bfloat16 else 1e-6   # grads stored in the logits dtype
    assert (g.float() - lf.grad).abs().max().item() < tol


@pytest.mark.parametrize("ls,z,cap", [(0.1, 0.0, 0.0), (0.0, 1e-4, 0.0), (0.0, 0.0, 30.0),
                                      (0.1, 1e-4, 30.0)])
def test_ce_kernel_smoothing_zloss_softcap(ls, z, cap):
    torch.manual_seed(1)
    T, V = 32, 1000
    logits = torch.randn(T, V, device=DEV) * 2
    tgt = torch.randint(0, V, (T,), device=DEV, dtype=torch.int32)
    loss, _ = kt.cross_entropy_fwd(logits, tgt, label_smoothing=ls, z_loss=z, softcap=cap)
    x = cap * torch.tanh(logits / cap) if cap > 0 else logits
    lse = x.logsumexp(-1)
    ref = (1 - ls) * (lse - x[torch.arange(T), tgt.long()])
    if ls > 0:
        ref = ref + ls * (lse - x.mean(-1))
    if z > 0:
        ref = ref + z * lse ** 2
    assert (loss - ref).abs().max().item() < 1e-4


def test_mixture_ce_fused_matches_fallback():
    """lm_mixture_loss takes the fused Triton path on CUDA; it must equal the chunked dense
    fallback in loss and in every grad (o_stack, w, tied classifier weight)."""
    import aum_ssm.training.losses as Lm

    torch.manual_seed(0)
    B, L, J1, d, V = 2, 257, 3, 64, 3001
    head = torch.nn.Linear(d, V, bias=False).to(DEV)
    o = (torch.randn(B, L, J1, d, device=DEV) * 0.1).requires_grad_()
    w = torch.softmax(torch.randn(B, L, J1, device=DEV), -1).requires_grad_()
    t = torch.randint(0, V, (B, L), device=DEV)
    assert Lm._fused_kernels(o.device) is not None

    mix = Lm.lm_mixture_loss(o, w, head, t)
    mix.backward()
    grads_fused = [x.grad.clone() for x in (o, w, head.weight)]
    for x in (o, w, head.weight):
        x.grad = None

    orig = Lm._fused_kernels
    Lm._fused_kernels = lambda dev_: None                  # force the chunked dense fallback
    try:
        ref = Lm.lm_mixture_loss(o, w, head, t)
        ref.backward()
    finally:
        Lm._fused_kernels = orig
    assert torch.allclose(mix, ref, rtol=1e-5)
    for a, b in zip(grads_fused, (o.grad, w.grad, head.weight.grad)):
        assert (a - b).abs().max().item() < 1e-4 * max(1e-8, b.abs().max().item()) + 1e-8
