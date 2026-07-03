"""Triton/CUDA fused silence block (kernels/triton) vs the in-repo oracles — the B1 acceptance
battery of AUM-nvidia-plan.md: forward trajectories vs silence_flat.flat_forward, gradients vs
autograd through the reference paths (flat + the per-token module loop, incl. the no_op
ablation and a segment-boundary-crossing L), and a full-model fused-vs-loop step. Skips
cleanly off-CUDA."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
kt = pytest.importorskip("kernels.triton")

from aum_ssm.modules.silence import SilenceBlock
from aum_ssm.ops.metal.silence_flat import flat_forward, precompute_streams
from aum_ssm.ops.metal.silence_metal import (build_stream_pack, build_weight_pack,
                                             silence_fused, unpack_save)

DEV = "cuda"
SLOTS = ("g_hat", "sigma_stack", "r_stack", "E", "pi", "w", "sigma_star")


def _inputs(B, L, H=8, Dh=64, grad=False, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda *s: torch.randn(*s, generator=g).to(DEV)  # noqa: E731
    d = dict(g=mk(B, L, 512), phi=mk(B, L, H).abs() * 3.0, m_t=mk(B, L, 32),
             s_t=mk(B, L, 1), alpha=torch.rand(B, L, H, generator=g).to(DEV) * 0.98,
             xw=mk(B, L, H, Dh) * 0.3, k_rot=mk(B, L, H, Dh) * 0.3)
    freqs = torch.logspace(0, -3, Dh // 2, device=DEV)
    halt_u = torch.rand(B, L, generator=g).to(DEV)
    if grad:
        for v in d.values():
            v.requires_grad_(True)
    return d, freqs, halt_u


def _silence(seed=42):
    torch.manual_seed(seed)
    return SilenceBlock(512).to(DEV).float()


def _projs(shapes, seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return {k: torch.randn(*s, generator=g).to(DEV) for k, s in shapes.items()}


def _slot_shapes(B, L):
    return {"g_hat": (B, L, 512), "sigma_stack": (B, L, 3, 128), "r_stack": (B, L, 3, 512),
            "E": (B, L, 3), "pi": (B, L), "w": (B, L, 3), "sigma_star": (B, L, 128)}


def _grads(module, leaves):
    out = {}
    for n, p in module.named_parameters():
        out[n] = None if p.grad is None else p.grad.detach().clone()
        p.grad = None
    for n, t in leaves.items():
        out["in." + n] = None if t.grad is None else t.grad.detach().clone()
        t.grad = None
    return out


def _assert_grads_close(ga, gb, rtol=3e-4, afloor=1e-6):
    """Relative max-err per tensor, with an ABSOLUTE noise floor for analytically-zero grads
    (plan §3.5): one-sided None means the other side must be ~0."""
    for n in ga:
        a, b = ga[n], gb[n]
        if a is None or b is None:
            other = a if b is None else b
            assert other is None or other.abs().max().item() < afloor, n
            continue
        scale = a.abs().max().item()
        err = (a - b).abs().max().item()
        assert err < max(afloor, rtol * scale), (n, err, scale)


@pytest.mark.parametrize("B,L,forced", [(2, 12, None), (2, 70, None), (2, 70, 1), (2, 33, 0)])
def test_fwd_matches_flat_forward(B, L, forced):
    # L=70 crosses the 64-token S-checkpoint segment boundary.
    silence = _silence()
    d, freqs, halt_u = _inputs(B, L)
    with torch.no_grad():
        ref = flat_forward(silence, precompute_streams(silence, d["g"], d["phi"], d["m_t"],
                                                       d["s_t"], freqs),
                           d["g"], d["m_t"], d["alpha"], d["xw"], d["k_rot"],
                           halt_u=None if forced is not None else halt_u,
                           forced_depth=forced)
        streams = build_stream_pack(silence, d["g"], d["phi"], d["m_t"], d["s_t"], freqs)
        save, j_star, S_final, _ = kt.aum_silence_fwd(
            streams, d["alpha"], d["xw"].reshape(B, L, -1), d["k_rot"].reshape(B, L, -1),
            halt_u, build_weight_pack(silence), silence.kappa,
            forced=-1 if forced is None else forced)
    out = unpack_save(save)
    for k in SLOTS:
        a, b = out[k], ref[k].reshape(out[k].shape)
        err = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
        assert err < 1e-5, (k, err)
    assert (j_star.long() == ref["j_star"]).all()
    assert torch.allclose(S_final, ref["S_final"], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("B,L,forced", [(2, 12, None), (2, 70, None), (2, 70, 1)])
def test_bwd_matches_flat_autograd(B, L, forced):
    silence = _silence()
    projs = _projs(_slot_shapes(B, L))
    loss_of = lambda s: sum((s[k] * projs[k]).sum() for k in projs)  # noqa: E731

    d, freqs, halt_u = _inputs(B, L, grad=True)
    hu = None if forced is not None else halt_u
    ref = flat_forward(silence, precompute_streams(silence, d["g"], d["phi"], d["m_t"],
                                                   d["s_t"], freqs),
                       d["g"], d["m_t"], d["alpha"], d["xw"], d["k_rot"],
                       halt_u=hu, forced_depth=forced)
    la = loss_of({k: ref[k].reshape(_slot_shapes(B, L)[k]) for k in projs})
    la.backward()
    ga = _grads(silence, d)

    o_t, aux = silence_fused(silence, d["g"], d["phi"], d["m_t"], d["s_t"], d["alpha"],
                             d["xw"], d["k_rot"], freqs, halt_u=hu, forced_depth=forced)
    slots = {"g_hat": aux.g_hat, "sigma_stack": torch.stack(aux.sigma_traj, 2),
             "r_stack": torch.stack(aux.r_traj, 2), "E": aux.E_traj, "pi": aux.pi,
             "w": aux.w, "sigma_star": aux.sigma_star}
    lb = loss_of(slots)
    lb.backward()
    gb = _grads(silence, d)

    assert abs(la.item() - lb.item()) <= 1e-5 * max(abs(la.item()), 1.0)
    _assert_grads_close(ga, gb)


def _module_loop(silence, d, freqs, halt_u, ablation=None, forced_depth=None):
    """The production per-token loop — AumBackbone._global_segment, replicated over
    SilenceBlock directly (same ops, same order)."""
    from aum_ssm.models.aum_lm import _cat_aux, _token_read
    g, phi, m_t, s_t = d["g"], d["phi"], d["m_t"], d["s_t"]
    alpha, xw, k_rot = d["alpha"], d["xw"], d["k_rot"]
    B, L, H, Dv = xw.shape
    S = xw.new_zeros(B, H, Dv, k_rot.shape[-1])
    sigma = g.new_zeros(B, 1, silence.d_sigma)
    phi_prev_t = phi.new_zeros(B, 1, H)
    outs, auxes = [], []
    for t in range(L):
        S_prev = S
        S = (alpha[:, t].unsqueeze(-1).unsqueeze(-1) * S_prev
             + xw[:, t].unsqueeze(-1) * k_rot[:, t].unsqueeze(-2))
        read_t = _token_read(S_prev, S, Dv, freqs)
        o_step, aux_t = silence(g[:, t:t + 1], read_t, phi[:, t:t + 1], phi_prev_t, sigma,
                                m_t[:, t:t + 1], s_t[:, t:t + 1], None, ablation,
                                forced_depth,
                                halt_u=None if halt_u is None else halt_u[:, t:t + 1])
        sigma = aux_t.sigma_star[:, :1]
        phi_prev_t = phi[:, t:t + 1]
        outs.append(o_step)
        auxes.append(aux_t)
    return torch.cat(outs, dim=1), _cat_aux(auxes)


@pytest.mark.parametrize("ablation,forced", [(None, None), ("no_op", None), (None, 1)])
def test_module_loop_parity(ablation, forced):
    """Fused route vs the per-token SilenceBlock loop: o_t + aux + gradients. Under no_op the
    candidates are identical, so some grads cancel analytically — the absolute noise floor in
    _assert_grads_close covers those (plan §3.5)."""
    B, L = 2, 70
    silence = _silence()
    silence.train()
    d, freqs, halt_u = _inputs(B, L, grad=True)
    hu = None if forced is not None else halt_u
    projs = _projs({**_slot_shapes(B, L), "o_t": (B, L, 512)})

    def loss_of(o_t, aux):
        slots = {"g_hat": aux.g_hat, "sigma_stack": torch.stack(aux.sigma_traj, 2),
                 "r_stack": torch.stack(aux.r_traj, 2), "E": aux.E_traj, "pi": aux.pi,
                 "w": aux.w, "sigma_star": aux.sigma_star, "o_t": o_t}
        return sum((slots[k] * projs[k]).sum() for k in projs)

    o_a, aux_a = _module_loop(silence, d, freqs, hu, ablation=ablation, forced_depth=forced)
    la = loss_of(o_a, aux_a)
    la.backward()
    ga = _grads(silence, d)

    o_b, aux_b = silence_fused(silence, d["g"], d["phi"], d["m_t"], d["s_t"], d["alpha"],
                               d["xw"], d["k_rot"], freqs, halt_u=hu, forced_depth=forced,
                               no_op=(ablation == "no_op"))
    assert (aux_a.j_star == aux_b.j_star).all()
    for name, a, b in [("o_t", o_a, o_b), ("w", aux_a.w, aux_b.w),
                       ("E", aux_a.E_traj, aux_b.E_traj),
                       ("sigma_star", aux_a.sigma_star, aux_b.sigma_star)]:
        err = (a - b).abs().max().item() / max(a.abs().max().item(), 1e-6)
        assert err < 1e-5, (name, err)
    lb = loss_of(o_b, aux_b)
    lb.backward()
    gb = _grads(silence, d)

    assert abs(la.item() - lb.item()) <= 1e-5 * max(abs(la.item()), 1.0)
    _assert_grads_close(ga, gb)


def test_full_model_fused_vs_loop():
    """Reference-geometry backbone, one projected-loss step: the fused CUDA route must match
    the reference loop in loss and in every parameter gradient."""
    from aum_ssm.models.aum_lm import AumLMHeadModel
    from aum_ssm.models.config_aum import AumConfig

    torch.manual_seed(0)
    cfg = AumConfig(n_layer=2, vocab_size=512, d_intermediate=128, silence_enabled=True)
    model = AumLMHeadModel(cfg, device=DEV).float().train()
    ids = torch.randint(0, 500, (2, 96), device=DEV)
    proj = torch.randn(512, device=DEV) / 512

    def run(fused):
        model.config.silence_fused = fused
        torch.manual_seed(1234)                    # identical halt_u draw on both routes
        o_t, aux = model.backbone(ids, return_aux=True)
        loss = (o_t * proj).sum() + aux.w.sum() * 0.1 + aux.E_traj.mean() + aux.pi.mean()
        loss.backward()
        g = {n: (None if p.grad is None else p.grad.detach().clone())
             for n, p in model.named_parameters()}
        for p in model.parameters():
            p.grad = None
        return loss.item(), g, aux.j_star.clone()

    loss_a, ga, js_a = run(fused=False)
    loss_b, gb, js_b = run(fused=True)
    model.config.silence_fused = True
    assert (js_a == js_b).all()
    assert abs(loss_a - loss_b) <= 1e-4 * max(abs(loss_a), 1.0), (loss_a, loss_b)
    _assert_grads_close(ga, gb, rtol=1e-3, afloor=1e-6)
