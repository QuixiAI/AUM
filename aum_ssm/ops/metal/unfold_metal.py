# AUM-Ø U-phase, Apple-Metal backend (§6). The SSD core runs as the `mamba2` Metal kernel on
# PyTorch's MPS stream via the SELF-CONTAINED vendored build in kernels/metal (no ThunderMittens
# dependency); the resonance rotary, k/v L2-norm, rho write-gate, cumlog, D-skip and gated readout
# are PyTorch ops around it. Validated against ssd_reference.
#
# mamba2(C,B,X,cumlog) computes ((C@Bᵀ) ⊙ exp(cumlog_i−cumlog_j) ⊙ causal) @ X, which is exactly the
# AUM readout S_t·R(φ)q with C=R(φ)q, B=k_rot=R(φ)·L2norm(k), X=ρτ·L2norm(v), cumlog=cumsum(−λτ).
# Headdim D=64 or D=128 (the Appendix-A reference is 4 heads x 128). Both the forward (mamba2) and
# the backward (mamba2_bwd → dC,dB,dX) run on the Metal GPU; dcumlog is the cheap host identity
# <dY,Y>−<dX,X>. `_ssd_core_ref` stays as the correctness oracle / fallback.

import os
import sys

import torch
from einops import rearrange

from aum_ssm.modules.ssd_reference import (
    aum_dynamics, _rotate_ladder, _l2norm, _gated_rmsnorm,
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
    """Forward + backward both on the Metal GPU (tk_torch.mamba2 / mamba2_bwd).

    dcumlog is the cheap host identity dcl_k = <dY_k,Y_k> − <dX_k,X_k> (rowsum/colsum of dM⊙M),
    so the kernels only emit dC/dB/dX. `_ssd_core_ref` remains the correctness oracle / fallback.
    """

    @staticmethod
    def forward(ctx, C, B, X, cumlog):
        km = _metal()
        with torch.no_grad():
            Y = km.mamba2(C.bfloat16(), B.bfloat16(), X.bfloat16(), cumlog.float()).to(C.dtype)
        ctx.save_for_backward(C, B, X, cumlog, Y)
        return Y

    @staticmethod
    def backward(ctx, dY):
        C, B, X, cumlog, Y = ctx.saved_tensors
        if _FUSED_BWD:
            dC, dB, dX = _metal().mamba2_bwd(C.bfloat16(), B.bfloat16(), X.bfloat16(),
                                             cumlog.float(), dY.bfloat16())
            dC, dB, dX = dC.float(), dB.float(), dX.float()
            dcumlog = (dY.float() * Y.float()).sum(-1) - (dX * X.float()).sum(-1)
        else:
            with torch.enable_grad():
                ins = [t.detach().requires_grad_(True) for t in (C, B, X, cumlog)]
                dC, dB, dX, dcumlog = torch.autograd.grad(_ssd_core_ref(*ins), ins, dY.float())
        return dC.to(C.dtype), dB.to(B.dtype), dX.to(X.dtype), dcumlog.to(cumlog.dtype)


def unfold_metal_chunk(q, k, v, tau_bar, lam_bar, r, theta, z=None, D=None,
                       dt_bias=0.0, eps=1e-4, norm_weight=None, freqs=None):
    """Metal U-phase readout h^U (§4). q,k,v: (B,L,H,headdim). Returns (B,L,H,headdim)."""
    assert q.shape[-1] in (64, 128), "aum mamba2 kernel supports headdim D=64 or D=128"
    assert q.shape[1] % 8 == 0, "mamba2 requires seqlen % 8 == 0"

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
