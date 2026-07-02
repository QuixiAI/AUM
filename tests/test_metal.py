"""Metal backend (M2/M3): the AUM U-phase via the self-contained kernels/metal build on MPS,
validated against the pure-PyTorch reference — at both headdim 64 and the Appendix-A headdim 128,
and end-to-end in the full model. Requires an Apple GPU + Xcode's Metal toolchain; skips otherwise."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
km = pytest.importorskip("kernels.metal")   # vendored, self-contained build (no ThunderMittens)

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


def _ssd_fwd_ref(C, B, X, cl):
    N = C.shape[-2]
    s = C.float() @ B.float().transpose(-1, -2)
    diff = cl[..., :, None] - cl[..., None, :]
    mask = torch.tril(torch.ones(N, N, device=C.device, dtype=torch.bool))
    # mask BEFORE exp: the strict upper triangle has large POSITIVE exponents at long N, and
    # exp -> inf would turn the masked product into NaN
    return (s * torch.exp(diff.masked_fill(~mask, float("-inf")))) @ X.float()


def _ssd_inputs(Bt, H, N, D, seed=0):
    torch.manual_seed(seed)
    C = torch.randn(Bt, H, N, D, device="mps") * 0.5
    Bm = torch.randn(Bt, H, N, D, device="mps") * 0.5
    X = torch.randn(Bt, H, N, D, device="mps")
    a = torch.sigmoid(torch.randn(Bt, H, N, device="mps")) * 0.5 + 0.5
    cl = torch.cumsum(torch.log(a), -1).float()
    return C, Bm, X, cl


@pytest.mark.parametrize("D", [64, 128])
def test_mamba2_bwd_kernel_matches_autograd(D):
    # the fused SSD backward kernel (dC,dB,dX + in-kernel dcumlog) vs autograd through the core
    Bt, H, N = 2, 4, 16
    C, Bm, X, cl = _ssd_inputs(Bt, H, N, D)
    dY = torch.randn(Bt, H, N, D, device="mps")
    Cr, Br, Xr, clr = (t.clone().detach().requires_grad_() for t in (C, Bm, X, cl))
    _ssd_fwd_ref(Cr, Br, Xr, clr).backward(dY.float())
    dC, dB, dX, dcl = km.mamba2_bwd(C.bfloat16(), Bm.bfloat16(), X.bfloat16(), cl, dY.bfloat16())
    torch.mps.synchronize()
    rel = lambda u, v: ((u - v).abs().max() / (v.abs().max() + 1e-6)).item()
    assert rel(dC.float(), Cr.grad) < 0.02, ("dC", rel(dC.float(), Cr.grad))
    assert rel(dB.float(), Br.grad) < 0.02, ("dB", rel(dB.float(), Br.grad))
    assert rel(dX.float(), Xr.grad) < 0.02, ("dX", rel(dX.float(), Xr.grad))
    assert rel(dcl.float(), clr.grad) < 0.02, ("dcumlog", rel(dcl.float(), clr.grad))


@pytest.mark.parametrize("N,D", [(128, 64), (256, 64), (128, 128), (256, 128)])
def test_ssd_chunked_bwd_matches_autograd(N, D):
    # the chunked linear-time backward (gradient states + reverse scan + chunk-bounded tiles,
    # dcl via the <dY,Y>-<dX,X> identity) vs autograd through the fp32 core
    C, Bm, X, cl = _ssd_inputs(2, 2, N, D, seed=5)
    dY = torch.randn(2, 2, N, D, device="mps")
    Cr, Br, Xr, clr = (t.clone().detach().requires_grad_() for t in (C, Bm, X, cl))
    _ssd_fwd_ref(Cr, Br, Xr, clr).backward(dY.float())
    dC, dB, dX, dcl = km.mamba2_bwd_chunked(C.bfloat16(), Bm.bfloat16(), X.bfloat16(),
                                            cl, dY.bfloat16())
    torch.mps.synchronize()
    rel = lambda u, v: ((u.float() - v).abs().max() / (v.abs().max() + 1e-6)).item()
    assert rel(dC, Cr.grad) < 0.03, ("dC", N, D, rel(dC, Cr.grad))
    assert rel(dB, Br.grad) < 0.03, ("dB", N, D, rel(dB, Br.grad))
    assert rel(dX, Xr.grad) < 0.03, ("dX", N, D, rel(dX, Xr.grad))
    assert rel(dcl, clr.grad) < 0.03, ("dcl", N, D, rel(dcl, clr.grad))


@pytest.mark.parametrize("bwd_fn", ["auto", "chunked"])
def test_mamba2_bwd_all_ones_decay(bwd_fn):
    """Degenerate decay (ported from ThunderMittens): a ≡ 1 so cumlog ≡ 0 and L ≡ 1 — no
    exponential taper anywhere, S = tril(C·Bᵀ). Stresses the raw row/col-sum and inter/intra
    dcl-split paths with no decay masking. Must still match the autograd oracle."""
    Bt, H, N, Dd = 1, 1, 128, 64
    torch.manual_seed(7)
    C = torch.randn(Bt, H, N, Dd, device="mps") * 0.5
    Bm = torch.randn(Bt, H, N, Dd, device="mps") * 0.5
    X = torch.randn(Bt, H, N, Dd, device="mps")
    cl = torch.zeros(Bt, H, N, device="mps")               # a == 1
    dY = torch.randn(Bt, H, N, Dd, device="mps")
    Cr, Br, Xr, clr = (t.clone().detach().requires_grad_() for t in (C, Bm, X, cl))
    _ssd_fwd_ref(Cr, Br, Xr, clr).backward(dY.float())
    fn = km.mamba2_bwd_chunked if bwd_fn == "chunked" else km.mamba2_bwd
    dC, dB, dX, dcl = fn(C.bfloat16(), Bm.bfloat16(), X.bfloat16(), cl, dY.bfloat16())
    torch.mps.synchronize()
    rel = lambda u, v: ((u.float() - v).abs().max() / (v.abs().max() + 1e-6)).item()
    for name, got, want in (("dC", dC, Cr.grad), ("dB", dB, Br.grad),
                            ("dX", dX, Xr.grad), ("dcl", dcl, clr.grad)):
        assert rel(got, want) < 0.03, (bwd_fn, name, rel(got, want))


@pytest.mark.parametrize("N,D,route", [
    (128, 64, "chunked"), (256, 64, "chunked"),     # forced linear-time (64x64 quadrant state)
    (128, 128, "chunked"), (256, 128, "chunked"),   # ... at the reference head dim
    (136, 64, "auto"), (136, 128, "auto"),          # ragged N -> quadratic
    (2048, 64, "auto"),                             # at/above the measured crossover -> chunked
])
def test_mamba2_forward_routes_match_reference(N, D, route):
    # Both routes (and the auto threshold) must agree with the fp32 reference.
    C, Bm, X, cl = _ssd_inputs(2, 2, N, D, seed=3)
    y_ref = _ssd_fwd_ref(C, Bm, X, cl)
    fn = km.mamba2_chunked if route == "chunked" else km.mamba2
    y = fn(C.bfloat16(), Bm.bfloat16(), X.bfloat16(), cl).float()
    torch.mps.synchronize()
    rel = ((y - y_ref).abs().max() / (y_ref.abs().max() + 1e-6)).item()
    assert rel < 0.03, (N, D, route, rel)


@pytest.mark.parametrize("H,D", [(8, 64), (4, 128)])
def test_fused_pipeline_forward_and_grads_match_reference(H, D):
    # the step-3 fully-fused path (aum_operands -> mamba2 -> aum_epilogue, recompute-based
    # backward) vs the fp32 reference, forward AND all gradients incl. dt_bias/D-skip/norm weight
    from aum_ssm.modules.ssd_reference import aum_unfold_chunk_ref
    torch.manual_seed(0)
    B, L = 2, 64
    mk = lambda *s: torch.randn(*s, device="mps")

    def build():
        torch.manual_seed(1)
        d = dict(q=mk(B, L, H, D), k=mk(B, L, H, D), v=mk(B, L, H, D), z=mk(B, L, H, D),
                 tau_bar=mk(B, L, H), lam_bar=mk(B, L, H), r=mk(B, L, H), theta=mk(B, L, H),
                 dt_bias=mk(H), D=mk(H, D), nw=mk(D))
        for t in d.values():
            t.requires_grad_(True)
        return d

    a, b = build(), build()
    # call the Function directly: the router sends grad-mode work to the composed path (the
    # fused backward is at parity until 3b), but its correctness must stay covered here
    from aum_ssm.ops.metal.unfold_metal import _AumUnfoldFused
    from aum_ssm.modules.ssd_reference import ladder_freqs
    out_m = _AumUnfoldFused.apply(a["q"], a["k"], a["v"], a["z"], a["tau_bar"], a["lam_bar"],
                                  a["r"], a["theta"], a["dt_bias"], a["D"], a["nw"],
                                  ladder_freqs(D // 2, device="mps"), 1e-4)
    out_r, _ = aum_unfold_chunk_ref(b["q"], b["k"], b["v"], b["tau_bar"], b["lam_bar"], b["r"],
                                    b["theta"], z=b["z"], D=b["D"], dt_bias=b["dt_bias"],
                                    norm_weight=b["nw"], block_len=16)
    torch.mps.synchronize()
    rel = lambda u, w: ((u.float() - w.float()).abs().max() / (w.float().abs().max() + 1e-6)).item()
    assert rel(out_m, out_r) < 0.04, rel(out_m, out_r)
    out_m.float().sum().backward()
    out_r.float().sum().backward()
    torch.mps.synchronize()
    for key in ("q", "k", "v", "z", "tau_bar", "theta", "dt_bias", "D", "nw"):
        rg = rel(a[key].grad, b[key].grad)
        assert rg < 0.08, (key, rg)


def test_fused_mixture_ce_matches_plain():
    # the §8 mixture LM loss via fused-linear-CE (never materializing (T,V) logits) vs the plain
    # einops path — loss AND grads for o_stack, w, and the tied classifier weight
    from aum_ssm.training.losses import lm_mixture_loss, _FusedMixtureCE
    torch.manual_seed(0)
    B, L, J1, d, V = 2, 16, 3, 32, 257
    head = torch.nn.Linear(d, V, bias=False).to("mps")

    def build():
        torch.manual_seed(1)
        o = torch.randn(B, L, J1, d, device="mps", requires_grad=True)
        wl = torch.rand(B, L, J1, device="mps")
        wl = (wl / wl.sum(-1, keepdim=True)).detach().requires_grad_(True)
        t = torch.randint(0, V, (B, L), device="mps")
        return o, wl, t

    o1, w1, t1 = build()
    loss_f = _FusedMixtureCE.apply(o1.reshape(-1, d), w1.reshape(-1), head.weight,
                                   t1.unsqueeze(-1).expand(B, L, J1).reshape(-1), B * L)
    loss_f.backward()
    gW1 = head.weight.grad.clone()
    head.weight.grad = None
    o2, w2, t2 = build()
    loss_p = lm_mixture_loss(o2, w2, torch.nn.Linear(d, V, bias=True).to("mps"), t2)  # bias -> plain
    # plain path with the SAME head weights (rebuild a bias-free plain computation manually)
    logits = o2.reshape(-1, d) @ head.weight.T
    ce = torch.nn.functional.cross_entropy(logits, t2.unsqueeze(-1).expand(B, L, J1).reshape(-1),
                                           reduction="none")
    loss_ref = (w2.reshape(-1) * ce).sum() / (B * L)
    loss_ref.backward()
    torch.mps.synchronize()
    rel = lambda u, v: ((u.float() - v.float()).abs().max() / (v.float().abs().max() + 1e-8)).item()
    assert abs(loss_f.item() - loss_ref.item()) < 2e-3 * max(1.0, abs(loss_ref.item()))
    assert rel(o1.grad, o2.grad) < 2e-2
    assert rel(w1.grad, w2.grad) < 2e-2
    assert rel(gW1, head.weight.grad) < 2e-2


@pytest.mark.parametrize("D", [64, 128])
def test_aum_decode_kernel_matches_math(D):
    # the fused single-token step S <- a*S + x⊗k_rot ; out = S·q_rot, directly vs the fp32 math
    torch.manual_seed(0)
    B, H = 2, 4
    S = torch.randn(B, H, D, D, device="mps")
    a = torch.rand(B, H, device="mps") * 0.5 + 0.5
    x = torch.randn(B, H, D, device="mps")
    k = torch.randn(B, H, D, device="mps")
    q = torch.randn(B, H, D, device="mps")
    S_ref = a[..., None, None] * S + x[..., :, None] * k[..., None, :]
    out_ref = torch.einsum("bhpn,bhn->bhp", S_ref, q)
    out, S_new = km.aum_decode(S.clone(), a, x, k, q)
    torch.mps.synchronize()
    assert float((S_new - S_ref).abs().max() / (S_ref.abs().max() + 1e-6)) < 1e-4
    assert float((out - out_ref).abs().max() / (out_ref.abs().max() + 1e-6)) < 1e-4


@pytest.mark.parametrize("H,D", [(8, 64), (4, 128)])
def test_metal_decode_step_matches_reference(H, D):
    # aum_unfold_step_metal (kernel core + PyTorch preamble) vs the pure-PyTorch step, carrying
    # (S, phi) across several tokens — the recurrent decode contract.
    from aum_ssm.ops.metal.unfold_metal import aum_unfold_step_metal
    from aum_ssm.modules.ssd_reference import aum_unfold_step_ref
    torch.manual_seed(0)
    B = 2
    mk = lambda *s: torch.randn(*s, device="mps")
    dt_bias, Dskip, nw = mk(H), mk(H, D), mk(D)
    Sm = torch.zeros(B, H, D, D, device="mps"); pm = torch.zeros(B, H, device="mps")
    Sr = torch.zeros(B, H, D, D, device="mps"); pr = torch.zeros(B, H, device="mps")
    for t in range(5):
        q, k, v, z = mk(B, 1, H, D), mk(B, 1, H, D), mk(B, 1, H, D), mk(B, 1, H, D)
        tb, lb, rr, th = mk(B, 1, H), mk(B, 1, H), mk(B, 1, H), mk(B, 1, H)
        kw = dict(z=z, D=Dskip, dt_bias=dt_bias, norm_weight=nw)
        hm, (Sm, pm) = aum_unfold_step_metal(q, k, v, tb, lb, rr, th, S0=Sm, phi0=pm, **kw)
        hr, (Sr, pr) = aum_unfold_step_ref(q, k, v, tb, lb, rr, th, S0=Sr, phi0=pr, **kw)
        torch.mps.synchronize()
        assert float((hm - hr).abs().max() / (hr.abs().max() + 1e-6)) < 1e-3, t


@pytest.mark.parametrize("silence", [False, True])
def test_metal_decode_matches_forward(silence):
    # end-to-end: decode with the metal step wired through the real model must reproduce the full
    # forward (the B1 decode oracle). Forward runs the fp32 reference so this isolates the step.
    # silence=True also exercises the metal step's swapped-query read_fn (the two-pass sigma carry).
    from aum_ssm.models.config_aum import AumConfig
    from aum_ssm.models.aum_lm import AumLMHeadModel
    from aum_ssm.utils.generation import InferenceParams
    torch.manual_seed(0)
    cfg = AumConfig(n_layer=2, vocab_size=128, d_intermediate=128, silence_enabled=silence)
    model = AumLMHeadModel(cfg).to("mps").eval()
    T, B, prompt = 16, 2, 8
    x = torch.randint(0, 128, (B, T), device="mps")
    for layer in model.backbone.layers:
        layer.unfold.kernel_backend = "reference"
    with torch.no_grad():
        ref = model(x).logits
        for layer in model.backbone.layers:
            layer.unfold.kernel_backend = "metal"
        ip = InferenceParams(max_seqlen=T, max_batch_size=B)
        ip.key_value_memory_dict = model.allocate_inference_cache(B, T)
        parts = [model(x[:, :prompt], inference_params=ip, num_last_tokens=1).logits]
        ip.seqlen_offset = prompt
        for t in range(prompt, T - 1):
            parts.append(model(x[:, t:t + 1], inference_params=ip, num_last_tokens=1).logits)
            ip.seqlen_offset += 1
        dec = torch.cat(parts, dim=1)
    torch.mps.synchronize()
    assert float((dec.float() - ref[:, prompt - 1:T - 1].float()).abs().max()) < 5e-3


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
        test_metal_decode_step_matches_reference(H, D)
    for D in (64, 128):
        test_aum_decode_kernel_matches_math(D)
    test_metal_decode_matches_forward()
    test_full_model_on_metal_backend()
    print("metal backend (fwd + bwd + decode, D=64 + D=128 + full model) OK")
