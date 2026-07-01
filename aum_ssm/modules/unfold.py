# Copyright (c) 2026.
# AUM-Ø U phase ("unfold"): resonant AFFINE evidence recurrence (§5-§6), Appendix-A layout.
#
# S_t = alpha_t S_{t-1} + rho_t tau_t (v_hat_t (x) k_rot_t);  h^U = silu(z) ⊙ RMSNorm(S_t q_rot + D v)
#
# Backend-pluggable: the recurrence is computed by one of {reference, metal, triton}. The
# `reference` backend (pure PyTorch, aum_ssm.modules.ssd_reference) runs on CPU/MPS/CUDA and is
# the correctness oracle; `metal` (ThunderMittens/tk_torch) and `triton` (NVIDIA, deferred) are
# imported lazily so importing this module never requires Triton. See AUM-metal-plan.md.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Unfold(nn.Module):
    def __init__(
        self,
        d_model,
        nheads=4,
        headdim=128,
        d_conv=4,
        conv_bias=True,
        chunk_size=64,
        m_dim=32,                     # precision-drive width consumed by the M phase
        kernel_backend="auto",        # auto|reference|metal|triton
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        eps=1e-4,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.nheads = nheads
        self.headdim = headdim
        self.d_inner = nheads * headdim
        self.d_conv = d_conv
        self.chunk_size = chunk_size
        self.m_dim = m_dim
        self.kernel_backend = kernel_backend
        self.eps = eps
        self.layer_idx = layer_idx
        d_controller = 128

        # Projections (Appendix A: unfold.*)
        self.controller = nn.Linear(d_model, d_controller, bias=False, **factory)        # W_c
        self.in_proj_qkv = nn.Linear(d_model, 3 * self.d_inner, bias=False, **factory)   # q,k,v
        self.in_proj_z = nn.Linear(d_model, self.d_inner, bias=False, **factory)         # output gate z
        # dyn = (tau_bar, lam_bar, r, theta) per head + m(m_dim) + s(1)
        self.in_proj_dyn = nn.Linear(d_controller, 4 * nheads + m_dim + 1, bias=False, **factory)

        # Short causal depthwise conv over q,k,v (Appendix A: unfold.conv1d [1536,1,4])
        self.conv1d = nn.Conv1d(
            3 * self.d_inner, 3 * self.d_inner, kernel_size=d_conv, groups=3 * self.d_inner,
            padding=d_conv - 1, bias=conv_bias, **factory,
        )

        # Per-head dynamics parameters
        dt = torch.exp(
            torch.rand(nheads, **factory) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)                          # b_tau, [nheads]
        self.dt_bias._no_weight_decay = True
        self.A_log = nn.Parameter(torch.zeros(nheads, **factory))    # per-head lambda base offset, [nheads]
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, **factory))   # per-channel skip, [d_inner]
        self.D._no_weight_decay = True

        # Gated-readout RMSNorm weight (Appendix A: unfold.norm [headdim])
        self.norm = _NormWeight(headdim, **factory)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory)

    def _dispatch(self, q, k, v, tau_bar, lam_bar, r, theta, z, D_hd):
        backend = self.kernel_backend
        if backend == "auto":
            backend = "reference"   # metal/triton wired in later milestones
        if backend == "reference":
            from aum_ssm.modules.ssd_reference import aum_unfold_chunk_ref
            L = q.shape[1]
            block_len = self.chunk_size if (L % self.chunk_size == 0) else L
            h, _ = aum_unfold_chunk_ref(
                q, k, v, tau_bar, lam_bar, r, theta, z=z, D=D_hd,
                dt_bias=self.dt_bias, eps=self.eps, block_len=block_len,
                norm_weight=self.norm.weight,
            )
            return h
        if backend == "metal":
            raise NotImplementedError("metal backend: see AUM-metal-plan.md (M2/M3)")
        if backend == "triton":
            raise NotImplementedError("triton backend deferred to NVIDIA bring-up")
        raise ValueError(f"unknown kernel_backend {backend!r}")

    def forward(self, x, inference_params=None, return_read=False, **kwargs):
        """x: (B, L, d_model) — already layer-normed by the evidence layer (§5 shared LN).

        Returns (out, m_t, s_t): out (B,L,d_model); m_t (B,L,m_dim) for the M phase;
        s_t (B,L,1) pressure drive for the top-layer silence block. When return_read=True,
        also returns (phi, read_fn) for the silence block's swapped-query reads (§4/§10).
        """
        if inference_params is not None:
            raise NotImplementedError("Unfold single-token decode is a later milestone (M2/Phase 3)")
        B, L, _ = x.shape

        qkv = self.in_proj_qkv(x)                                    # (B,L,3*d_inner)
        qkv = F.silu(self.conv1d(qkv.transpose(1, 2))[..., :L].transpose(1, 2))
        q, k, v = qkv.split(self.d_inner, dim=-1)
        q = rearrange(q, "b l (h p) -> b l h p", p=self.headdim)
        k = rearrange(k, "b l (h p) -> b l h p", p=self.headdim)
        v = rearrange(v, "b l (h p) -> b l h p", p=self.headdim)
        z = rearrange(self.in_proj_z(x), "b l (h p) -> b l h p", p=self.headdim)

        c = self.controller(x)
        dyn = self.in_proj_dyn(c)
        heads = dyn[..., : 4 * self.nheads].reshape(B, L, self.nheads, 4)
        tau_bar, lam_bar, r, theta = heads.unbind(dim=-1)           # each (B,L,nheads)
        m_t = dyn[..., 4 * self.nheads : 4 * self.nheads + self.m_dim]   # (B,L,m_dim)
        s_t = dyn[..., 4 * self.nheads + self.m_dim :]                   # (B,L,1)

        lam_bar = lam_bar + self.A_log                              # per-head lambda base offset (effective lambda_bar)
        D_hd = self.D.view(self.nheads, self.headdim)

        h = self._dispatch(q, k, v, tau_bar, lam_bar, r, theta, z, D_hd)  # (B,L,nheads,headdim)
        out = self.out_proj(rearrange(h, "b l h p -> b l (h p)"))
        if not return_read:
            return out, m_t, s_t

        # Bind the swapped-query read against THIS layer's evidence state (§4/§10). The read is a
        # linear-attention readout with the same write (k, v, dynamics); only the query changes.
        from aum_ssm.modules.ssd_reference import aum_dynamics, aum_state_readout_ref
        _, _, _, dphi = aum_dynamics(tau_bar, lam_bar, r, theta, self.dt_bias, self.eps)
        phi = torch.cumsum(dphi, dim=1)                             # (B,L,nheads) resonance phase
        k_c, v_c, tb, lb, rw, th = k, v, tau_bar, lam_bar, r, theta

        def read_fn(query, phi_arg=None, exclude_current=False):
            L_ = query.shape[1]
            bl = self.chunk_size if (L_ % self.chunk_size == 0) else L_
            qh = rearrange(query, "b l (h p) -> b l h p", p=self.headdim)
            rr = aum_state_readout_ref(qh, k_c, v_c, tb, lb, rw, th, phi=phi, dt_bias=self.dt_bias,
                                       eps=self.eps, block_len=bl, exclude_current=exclude_current)
            return rearrange(rr, "b l h p -> b l (h p)")

        return out, m_t, s_t, phi, read_fn

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError("Unfold decode cache is a later milestone (M2/Phase 3)")


class _NormWeight(nn.Module):
    """Holds the gated-readout RMSNorm weight (state-dict key `norm.weight`, [headdim])."""

    def __init__(self, dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
