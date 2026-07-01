# AUM-Ø U-phase, Apple-Metal backend (§6). The SSD core runs as ThunderMittens' `mamba2` Metal
# kernel on PyTorch's MPS stream (via tk_torch); the resonance rotary, k/v L2-norm, rho write-gate,
# cumlog, D-skip and gated readout are PyTorch ops around it. Validated against ssd_reference.
#
# tk_torch.mamba2(C,B,X,cumlog) computes ((C@Bᵀ) ⊙ exp(cumlog_i−cumlog_j) ⊙ causal) @ X, which is
# exactly the AUM readout S_t·R(φ)q with C=R(φ)q, B=k_rot=R(φ)·L2norm(k), X=ρτ·L2norm(v),
# cumlog=cumsum(−λτ). The kernel is D=64 and forward-only, so:
#   - headdim must be 64 (headdim-128 needs a D=128 mamba2 instantiation in ThunderMittens — M5),
#   - the SSD core is wrapped in a torch.autograd.Function whose backward recomputes the (cheap,
#     O(N²)) SSD gradient in PyTorch — a correct Stage-1 fusion; a fused Metal backward is M4.
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


class _Mamba2SSD(torch.autograd.Function):
    """Forward: tk_torch.mamba2 Metal kernel. Backward: the reference SSD core (PyTorch)."""

    @staticmethod
    def forward(ctx, C, B, X, cumlog):
        import tk_torch
        ctx.save_for_backward(C, B, X, cumlog)
        with torch.no_grad():
            Y = tk_torch.mamba2(C.bfloat16(), B.bfloat16(), X.bfloat16(), cumlog.float())
        return Y.to(C.dtype)

    @staticmethod
    def backward(ctx, dY):
        C, B, X, cumlog = ctx.saved_tensors
        with torch.enable_grad():
            ins = [t.detach().requires_grad_(True) for t in (C, B, X, cumlog)]
            Y = _ssd_core_ref(*ins)
            grads = torch.autograd.grad(Y, ins, dY.float())
        return tuple(g.to(t.dtype) for g, t in zip(grads, (C, B, X, cumlog)))


def unfold_metal_chunk(q, k, v, tau_bar, lam_bar, r, theta, z=None, D=None,
                       dt_bias=0.0, eps=1e-4, norm_weight=None):
    """Metal U-phase readout h^U (§6). q,k,v: (B,L,H,headdim=64). Returns (B,L,H,headdim)."""
    assert q.shape[-1] == 64, "tk_torch.mamba2 is D=64; headdim-128 needs a D=128 TM instantiation (M5)"
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
