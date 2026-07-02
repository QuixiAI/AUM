# AUM-Ø U-phase, Apple-Metal backend (§6). The SSD core runs as the `mamba2` Metal kernel on
# PyTorch's MPS stream via the SELF-CONTAINED vendored build in kernels/metal (no ThunderMittens
# dependency); the resonance rotary, k/v L2-norm, rho write-gate, cumlog, D-skip and gated readout
# are PyTorch ops around it. Validated against ssd_reference.
#
# mamba2(C,B,X,cumlog) computes ((C@Bᵀ) ⊙ exp(cumlog_i−cumlog_j) ⊙ causal) @ X, which is exactly the
# AUM readout S_t·R(φ)q with C=R(φ)q, B=k_rot=R(φ)·L2norm(k), X=ρτ·L2norm(v), cumlog=cumsum(−λτ).
# Headdim D=64 or D=128 (the Appendix-A reference is 4 heads x 128). The forward auto-routes to a
# chunked LINEAR-TIME 3-kernel pipeline at D=64 (quadratic materialized form otherwise); the
# backward (mamba2_bwd) returns dC,dB,dX and an fp32 in-kernel dcumlog = rowsum(M)−colsum(M).
# `_ssd_core_ref` stays as the correctness oracle / fallback.

import os
import sys

import torch
from einops import rearrange

from aum_ssm.modules.ssd_reference import (
    aum_dynamics, ladder_freqs, _rotate_ladder, _l2norm, _gated_rmsnorm,
)


def _metal():
    """Import the vendored, self-contained Metal build (kernels/metal) — builds on first use."""
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import kernels.metal as km
    return km


def _ssd_core_ref(C, B, X, cumlog):
    """The mamba2 numerator in PyTorch (the backward oracle): ((C@Bᵀ)⊙decay⊙causal) @ X."""
    N = C.shape[-2]
    scores = C.float() @ B.float().transpose(-1, -2)
    decay = torch.exp(cumlog[..., :, None] - cumlog[..., None, :])
    mask = torch.tril(torch.ones(N, N, device=C.device, dtype=torch.float32))
    return (scores * decay * mask) @ X.float()


_FUSED_BWD = True   # M4: fused mamba2_bwd Metal kernel. Set False to fall back to the PyTorch oracle.


class _Mamba2SSD(torch.autograd.Function):
    """Forward + backward both on the Metal GPU (kernels.metal mamba2 / mamba2_bwd).

    The forward auto-routes to the chunked LINEAR-TIME pipeline at D=64 (quadratic otherwise).
    The backward kernels emit dC/dB/dX AND dcumlog = rowsum(M)−colsum(M) with fp32 in-kernel
    accumulation, so the forward output Y is never saved. `_ssd_core_ref` remains the
    correctness oracle / fallback.
    """

    @staticmethod
    def forward(ctx, C, B, X, cumlog):
        km = _metal()
        with torch.no_grad():
            Y = km.mamba2(C.bfloat16(), B.bfloat16(), X.bfloat16(), cumlog.float()).to(C.dtype)
        ctx.save_for_backward(C, B, X, cumlog)
        return Y

    @staticmethod
    def backward(ctx, dY):
        C, B, X, cumlog = ctx.saved_tensors
        if _FUSED_BWD:
            dC, dB, dX, dcumlog = _metal().mamba2_bwd(C.bfloat16(), B.bfloat16(), X.bfloat16(),
                                                      cumlog.float(), dY.bfloat16())
        else:
            with torch.enable_grad():
                ins = [t.detach().requires_grad_(True) for t in (C, B, X, cumlog)]
                dC, dB, dX, dcumlog = torch.autograd.grad(_ssd_core_ref(*ins), ins, dY.float())
        return dC.to(C.dtype), dB.to(B.dtype), dX.to(X.dtype), dcumlog.to(cumlog.dtype)


class _AumUnfoldFused(torch.autograd.Function):
    """The fully-fused U-phase (§4): THREE kernel passes replace the whole host chain.

    forward:  aum_operands (rotation ladder + k/v L2-norms + rho*tau + layout transpose, one pass)
              -> mamba2 (auto-routed cooperative chunked/quadratic SSD)
              -> aum_epilogue (D-skip + gated RMSNorm + transpose back, one pass).
    backward: recompute-based — the pre/postamble subgraphs are rebuilt under autograd on the
              saved RAW inputs (exact math, small graphs), with the SSD gradient in the middle
              computed by the linear-time mamba2_bwd kernels. Only the raw inputs + Y are saved;
              C/B/X are recomputed by the operands kernel (cheap, memory win).
    """

    @staticmethod
    def forward(ctx, q, k, v, z, tau_bar, lam_bar, r, theta, dt_bias, D_hd, norm_w, freqs, eps):
        km = _metal()
        H, Dh = q.shape[2], q.shape[3]
        with torch.no_grad():
            tau, alpha_log, rho, dphi = aum_dynamics(tau_bar, lam_bar, r, theta, dt_bias, eps)
            phi = torch.cumsum(dphi, dim=1)                     # (B,L,H) fp32 smalls
            cumlog = rearrange(torch.cumsum(alpha_log, dim=1), "b l h -> b h l").contiguous()
            C, Bm, Xm = km.aum_operands(q.bfloat16(), k.bfloat16(), v.bfloat16(),
                                        phi.float().contiguous(),
                                        (rho * tau).float().contiguous(),
                                        freqs.float(), 1e-6)
            Y = km.mamba2(C, Bm, Xm, cumlog.float())
            d_flat = (D_hd if D_hd.dim() == 2
                      else D_hd.unsqueeze(-1).expand(H, Dh)).reshape(-1).float().contiguous()
            out = km.aum_epilogue(Y, v.bfloat16(), z.bfloat16(), d_flat,
                                  norm_w.float().contiguous(), 1e-5)
        ctx.save_for_backward(q, k, v, z, tau_bar, lam_bar, r, theta, dt_bias, D_hd, norm_w,
                              freqs, Y)
        ctx.eps = eps
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, dout):
        km = _metal()
        (q, k, v, z, tau_bar, lam_bar, r, theta, dt_bias, D_hd, norm_w, freqs, Y) = ctx.saved_tensors
        eps = ctx.eps

        # ---- epilogue backward (recompute under autograd; exact) ----
        Y_r = rearrange(Y.float(), "b h l d -> b l h d").detach().requires_grad_(True)
        v1 = v.detach().requires_grad_(True)
        z1 = z.detach().requires_grad_(True)
        D1 = D_hd.detach().requires_grad_(True)
        w1 = norm_w.detach().requires_grad_(True)
        with torch.enable_grad():
            y_skip = Y_r + (D1 if D1.dim() == 2 else D1.unsqueeze(-1)) * v1
            out2 = _gated_rmsnorm(y_skip, z1, w1)
        dY_r, dv_post, dz, dD, dw = torch.autograd.grad(out2, (Y_r, v1, z1, D1, w1), dout.float())

        # ---- preamble recompute (one graph; its values also feed the SSD backward kernels) ----
        q2 = q.detach().requires_grad_(True)
        k2 = k.detach().requires_grad_(True)
        v2 = v.detach().requires_grad_(True)
        tb2 = tau_bar.detach().requires_grad_(True)
        lb2 = lam_bar.detach().requires_grad_(True)
        r2 = r.detach().requires_grad_(True)
        th2 = theta.detach().requires_grad_(True)
        db2 = dt_bias.detach().requires_grad_(True)
        with torch.enable_grad():
            tau_, alog_, rho_, dphi_ = aum_dynamics(tb2, lb2, r2, th2, db2, eps)
            phi_ = torch.cumsum(dphi_, dim=1)
            C2 = _rotate_ladder(q2, phi_, freqs)
            B2 = _rotate_ladder(_l2norm(k2), phi_, freqs)
            X2 = (rho_ * tau_).unsqueeze(-1) * _l2norm(v2)
            cl2 = rearrange(torch.cumsum(alog_, dim=1), "b l h -> b h l")

        # ---- SSD backward (linear-time kernels) on the recomputed operand values ----
        with torch.no_grad():
            to_k = lambda t: rearrange(t.detach(), "b l h d -> b h l d").contiguous().bfloat16()
            dY = rearrange(dY_r, "b l h d -> b h l d").contiguous().bfloat16()
            dC, dB, dX, dcl = km.mamba2_bwd(to_k(C2), to_k(B2), to_k(X2),
                                            cl2.detach().contiguous().float(), dY)
        gdC = rearrange(dC.float(), "b h l d -> b l h d")
        gdB = rearrange(dB.float(), "b h l d -> b l h d")
        gdX = rearrange(dX.float(), "b h l d -> b l h d")
        dq, dk, dv_pre, dtb, dlb, dr, dth, ddb = torch.autograd.grad(
            (C2, B2, X2, cl2), (q2, k2, v2, tb2, lb2, r2, th2, db2),
            (gdC, gdB, gdX, dcl.float()))

        return (dq.to(q.dtype), dk.to(k.dtype), (dv_pre + dv_post).to(v.dtype), dz.to(z.dtype),
                dtb.to(tau_bar.dtype), dlb.to(lam_bar.dtype), dr.to(r.dtype), dth.to(theta.dtype),
                ddb.to(dt_bias.dtype), dD.to(D_hd.dtype), dw.to(norm_w.dtype), None, None)


_FUSED_PIPELINE = True   # step-3 fused path; False falls back to the composed host chain


def unfold_metal_chunk(q, k, v, tau_bar, lam_bar, r, theta, z=None, D=None,
                       dt_bias=0.0, eps=1e-4, norm_weight=None, freqs=None):
    """Metal U-phase readout h^U (§4). q,k,v: (B,L,H,headdim). Returns (B,L,H,headdim)."""
    assert q.shape[-1] in (64, 128), "aum mamba2 kernel supports headdim D=64 or D=128"
    assert q.shape[1] % 8 == 0, "mamba2 requires seqlen % 8 == 0"

    # Fused route: 2.2-4x on forward-only work (inference / prefill / eval — the host
    # preamble+postamble collapse into 3 kernel passes). Training currently stays on the
    # composed path: the fused backward is CORRECT (grad-tested) but its recompute strategy
    # is at parity with composed — the win there needs the backward-side fused kernels (3b).
    needs_grad = torch.is_grad_enabled() and any(
        torch.is_tensor(t) and t.requires_grad
        for t in (q, k, v, z, tau_bar, lam_bar, r, theta, dt_bias, D, norm_weight))
    if _FUSED_PIPELINE and not needs_grad and z is not None and D is not None \
            and torch.is_tensor(dt_bias):
        H, Dh = q.shape[2], q.shape[3]
        fr = freqs if freqs is not None else ladder_freqs(Dh // 2, device=q.device)
        nw = norm_weight if norm_weight is not None else torch.ones(Dh, device=q.device)
        return _AumUnfoldFused.apply(q, k, v, z, tau_bar, lam_bar, r, theta, dt_bias, D, nw,
                                     fr, eps)

    # composed fallback (also the oracle path for z/D-less configs)
    tau, alpha_log, rho, dphi = aum_dynamics(tau_bar, lam_bar, r, theta, dt_bias, eps)
    phi = torch.cumsum(dphi, dim=1)
    C = rearrange(_rotate_ladder(q, phi, freqs), "b l h d -> b h l d")
    Bm = rearrange(_rotate_ladder(_l2norm(k), phi, freqs), "b l h d -> b h l d")
    Xm = rearrange((rho * tau).unsqueeze(-1) * _l2norm(v), "b l h d -> b h l d")
    cumlog = rearrange(torch.cumsum(alpha_log, dim=1), "b l h -> b h l")

    Y = _Mamba2SSD.apply(C.contiguous(), Bm.contiguous(), Xm.contiguous(), cumlog.contiguous())
    Y = rearrange(Y, "b h l d -> b l h d")
    if D is not None:
        Y = Y + (D if D.dim() == 2 else D.unsqueeze(-1)) * v
    if z is not None:
        Y = _gated_rmsnorm(Y, z, norm_weight)
    return Y


def aum_unfold_step_metal(q, k, v, tau_bar, lam_bar, r, theta, z=None, D=None,
                          dt_bias=0.0, eps=1e-4, S0=None, phi0=None, norm_weight=None, freqs=None):
    """Metal single-token decode step (§6) — a drop-in for `aum_unfold_step_ref`.

    The `aum_decode` kernel does the D×D state update + readout (S <- alpha*S + x⊗k_rot; out =
    S·q_rot); the dynamics, rotary, k/v L2-norm, D-skip and gated-RMSNorm are PyTorch, exactly as
    the chunk path splits the work around `mamba2`. q,k,v: (B,1,H,D). Returns h_U (B,1,H,D) and the
    updated (S (B,H,Dv,Dqk) fp32, phi (B,H))."""
    km = _metal()
    B, L, H, Dqk = q.shape
    assert L == 1, "aum_unfold_step_metal is a single-token step"
    assert Dqk in (64, 128), "aum_decode supports headdim D=64 or D=128"
    Dv = v.shape[-1]
    device, dtype = q.device, q.dtype
    S = (torch.zeros(B, H, Dv, Dqk, dtype=torch.float32, device=device)
         if S0 is None else S0.clone().to(torch.float32))
    phi_prev = (torch.zeros(B, H, dtype=torch.float32, device=device)
                if phi0 is None else phi0.to(torch.float32))

    tau, alpha_log, rho, dphi = aum_dynamics(
        tau_bar[:, 0], lam_bar[:, 0], r[:, 0], theta[:, 0], dt_bias, eps)
    phi = phi_prev + dphi                                   # (B,H)
    q_rot = _rotate_ladder(q[:, 0], phi, freqs)             # (B,H,Dqk)
    k_rot = _rotate_ladder(_l2norm(k[:, 0]), phi, freqs)    # (B,H,Dqk)
    x = (rho * tau).unsqueeze(-1) * _l2norm(v[:, 0])        # (B,H,Dv)  = rho*tau*v_hat
    alpha = torch.exp(alpha_log)                            # (B,H)

    f = lambda t: t.float().contiguous()
    out_core, S_new = km.aum_decode(S.contiguous(), f(alpha), f(x), f(k_rot), f(q_rot))

    out = out_core.to(dtype)                                # (B,H,Dv)
    if D is not None:
        out = out + (D if D.dim() == 2 else D.unsqueeze(-1)) * v[:, 0]
    h = out.unsqueeze(1)                                    # (B,1,H,Dv)
    if z is not None:
        h = _gated_rmsnorm(h, z, norm_weight)
    return h, (S_new, phi)
