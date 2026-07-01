# AUM-Ø U-phase, Apple-Metal backend (§6). The SSD core runs as ThunderMittens' `mamba2` Metal
# kernel on PyTorch's MPS stream (via tk_torch); the resonance rotary, k/v L2-norm, rho write-gate,
# cumlog, D-skip and gated readout are PyTorch ops around it. Validated against ssd_reference.
#
# tk_torch.mamba2(C,B,X,cumlog) computes ((C@Bᵀ) ⊙ exp(cumlog_i−cumlog_j) ⊙ causal) @ X, which is
# exactly the AUM readout S_t·R(φ)q with C=R(φ)q, B=k_rot=R(φ)·L2norm(k), X=ρτ·L2norm(v),
# cumlog=cumsum(−λτ). Headdim D=64 or D=128 (the Appendix-A reference is 4 heads x 128). Both the
# forward (mamba2) and the backward (mamba2_bwd → dC,dB,dX) run on the Metal GPU; dcumlog is the
# cheap host identity <dY,Y>−<dX,X>. `_ssd_core_ref` stays as the correctness oracle / fallback.
#
# tk_torch lives in the ThunderMittens repo; put .../ThunderMittens/kernels on PYTHONPATH.

import torch
from einops import rearrange

from aum_ssm.modules.ssd_reference import (
    aum_dynamics, _rotate_single_phase, _l2norm, _gated_rmsnorm,
)


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
        import tk_torch
        with torch.no_grad():
            Y = tk_torch.mamba2(C.bfloat16(), B.bfloat16(), X.bfloat16(), cumlog.float()).to(C.dtype)
        ctx.save_for_backward(C, B, X, cumlog, Y)
        return Y

    @staticmethod
    def backward(ctx, dY):
        C, B, X, cumlog, Y = ctx.saved_tensors
        if _FUSED_BWD:
            import tk_torch
            dC, dB, dX = tk_torch.mamba2_bwd(C.bfloat16(), B.bfloat16(), X.bfloat16(),
                                             cumlog.float(), dY.bfloat16())
            dC, dB, dX = dC.float(), dB.float(), dX.float()
            dcumlog = (dY.float() * Y.float()).sum(-1) - (dX * X.float()).sum(-1)
        else:
            with torch.enable_grad():
                ins = [t.detach().requires_grad_(True) for t in (C, B, X, cumlog)]
                dC, dB, dX, dcumlog = torch.autograd.grad(_ssd_core_ref(*ins), ins, dY.float())
        return dC.to(C.dtype), dB.to(B.dtype), dX.to(X.dtype), dcumlog.to(cumlog.dtype)


def unfold_metal_chunk(q, k, v, tau_bar, lam_bar, r, theta, z=None, D=None,
                       dt_bias=0.0, eps=1e-4, norm_weight=None):
    """Metal U-phase readout h^U (§6). q,k,v: (B,L,H,headdim=64). Returns (B,L,H,headdim)."""
    assert q.shape[-1] in (64, 128), "tk_torch.mamba2 supports headdim D=64 or D=128"
    assert q.shape[1] % 8 == 0, "mamba2 requires seqlen % 8 == 0"

    tau, alpha_log, rho, dphi = aum_dynamics(tau_bar, lam_bar, r, theta, dt_bias, eps)
    phi = torch.cumsum(dphi, dim=1)
    C = rearrange(_rotate_single_phase(q, phi), "b l h d -> b h l d")
    Bm = rearrange(_rotate_single_phase(_l2norm(k), phi), "b l h d -> b h l d")
    Xm = rearrange((rho * tau).unsqueeze(-1) * _l2norm(v), "b l h d -> b h l d")
    cumlog = rearrange(torch.cumsum(alpha_log, dim=1), "b l h -> b h l")

    Y = _Mamba2SSD.apply(C.contiguous(), Bm.contiguous(), Xm.contiguous(), cumlog.contiguous())
    Y = rearrange(Y, "b h l d -> b l h d")
    if D is not None:
        Y = Y + (D if D.dim() == 2 else D.unsqueeze(-1)) * v
    if z is not None:
        Y = _gated_rmsnorm(Y, z, norm_weight)
    return Y
