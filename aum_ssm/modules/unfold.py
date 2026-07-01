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
            from aum_ssm.ops.metal.unfold_metal import unfold_metal_chunk
            return unfold_metal_chunk(q, k, v, tau_bar, lam_bar, r, theta, z=z, D=D_hd,
                                      dt_bias=self.dt_bias, eps=self.eps, norm_weight=self.norm.weight)
        if backend == "triton":
            raise NotImplementedError("triton backend deferred to NVIDIA bring-up")
        raise ValueError(f"unknown kernel_backend {backend!r}")

    def forward(self, x, inference_params=None, cache=None, return_read=False, **kwargs):
        """x: (B, L, d_model) — already layer-normed by the evidence layer (§5 shared LN).

        Returns (out, m_t, s_t) [+ (phi, read_fn) when return_read]. Handles three modes:
        training (chunk kernel), prefill (chunk ref + capture S_T,phi_T,conv), and single-token
        decode (recurrent step via aum_unfold_step_ref). `cache` = (conv_state, S, phi) is supplied
        by the evidence layer; decode always uses the reference step.
        """
        from aum_ssm.modules.ssd_reference import (
            aum_dynamics, aum_state_readout_ref, aum_unfold_chunk_ref, aum_unfold_step_ref,
            _rotate_single_phase,
        )
        B, L, _ = x.shape
        decoding = inference_params is not None and inference_params.seqlen_offset > 0
        conv_state = S_cache = phi_cache = None
        if cache is not None:
            conv_state, S_cache, phi_cache = cache

        # ---- projections (shared) + conv (branch on decode) ----
        pre = self.in_proj_qkv(x)                                   # (B,L,3*d_inner)
        if decoding:
            qkv_c, new_conv = self._conv_step(pre[:, 0], conv_state)
            conv_state.copy_(new_conv)
            qkv = F.silu(qkv_c).unsqueeze(1)
        else:
            qkv = F.silu(self.conv1d(pre.transpose(1, 2))[..., :L].transpose(1, 2))
            if conv_state is not None:                             # prefill: seed the conv state
                cs = pre.transpose(1, 2)
                conv_state.copy_(cs[..., -(self.d_conv - 1):] if L >= self.d_conv - 1
                                 else F.pad(cs, (self.d_conv - 1 - L, 0)))
        q, k, v = qkv.split(self.d_inner, dim=-1)
        q = rearrange(q, "b l (h p) -> b l h p", p=self.headdim)
        k = rearrange(k, "b l (h p) -> b l h p", p=self.headdim)
        v = rearrange(v, "b l (h p) -> b l h p", p=self.headdim)
        z = rearrange(self.in_proj_z(x), "b l (h p) -> b l h p", p=self.headdim)

        dyn = self.in_proj_dyn(self.controller(x))
        heads = dyn[..., : 4 * self.nheads].reshape(B, L, self.nheads, 4)
        tau_bar, lam_bar, r, theta = heads.unbind(dim=-1)
        m_t = dyn[..., 4 * self.nheads : 4 * self.nheads + self.m_dim]
        s_t = dyn[..., 4 * self.nheads + self.m_dim :]
        lam_bar = lam_bar + self.A_log
        D_hd = self.D.view(self.nheads, self.headdim)
        nw = self.norm.weight

        # ---- recurrence ----
        if decoding:
            S_pre = S_cache.clone()                                # S_{t-1} for the predictive read
            step = aum_unfold_step_ref
            if self.kernel_backend == "metal":                     # fused D×D core on the Metal GPU
                from aum_ssm.ops.metal.unfold_metal import aum_unfold_step_metal
                step = aum_unfold_step_metal
            h, (S_new, phi_new) = step(
                q, k, v, tau_bar, lam_bar, r, theta, z=z, D=D_hd, dt_bias=self.dt_bias,
                eps=self.eps, S0=S_cache, phi0=phi_cache, norm_weight=nw)
            S_cache.copy_(S_new); phi_cache.copy_(phi_new)
            out = self.out_proj(rearrange(h, "b l h p -> b l (h p)"))
            if not return_read:
                return out, m_t, s_t
            phi_1H = phi_new                                       # (B,H) current phase

            def read_fn(query, phi_arg=None, exclude_current=False):
                S = S_pre if exclude_current else S_new
                qh = rearrange(query, "b l (h p) -> b l h p", p=self.headdim)
                q_rot = _rotate_single_phase(qh, phi_1H.unsqueeze(1))
                rr = torch.einsum("bhpn,blhn->blhp", S, q_rot)
                return rearrange(rr, "b l h p -> b l (h p)")

            return out, m_t, s_t, phi_new.unsqueeze(1), read_fn

        if cache is not None:                                     # prefill: capture final state
            bl = self.chunk_size if (L % self.chunk_size == 0) else L
            h, (S_T, phi_T) = aum_unfold_chunk_ref(
                q, k, v, tau_bar, lam_bar, r, theta, z=z, D=D_hd, dt_bias=self.dt_bias,
                eps=self.eps, block_len=bl, norm_weight=nw)
            S_cache.copy_(S_T); phi_cache.copy_(phi_T)
        else:                                                     # training
            h = self._dispatch(q, k, v, tau_bar, lam_bar, r, theta, z, D_hd)
        out = self.out_proj(rearrange(h, "b l h p -> b l (h p)"))
        if not return_read:
            return out, m_t, s_t

        _, _, _, dphi = aum_dynamics(tau_bar, lam_bar, r, theta, self.dt_bias, self.eps)
        phi = torch.cumsum(dphi, dim=1)
        k_c, v_c, tb, lb, rw, th = k, v, tau_bar, lam_bar, r, theta

        def read_fn(query, phi_arg=None, exclude_current=False):
            L_ = query.shape[1]
            bl_ = self.chunk_size if (L_ % self.chunk_size == 0) else L_
            qh = rearrange(query, "b l (h p) -> b l h p", p=self.headdim)
            rr = aum_state_readout_ref(qh, k_c, v_c, tb, lb, rw, th, phi=phi, dt_bias=self.dt_bias,
                                       eps=self.eps, block_len=bl_, exclude_current=exclude_current)
            return rearrange(rr, "b l h p -> b l (h p)")

        return out, m_t, s_t, phi, read_fn

    def _conv_step(self, x_new, conv_state):
        """Single-token causal depthwise conv. x_new (B,3*d_inner); conv_state (B,3*d_inner,d_conv-1)."""
        w = self.conv1d.weight.squeeze(1)                          # (3*d_inner, d_conv)
        window = torch.cat([conv_state, x_new.unsqueeze(-1)], dim=-1)   # (B,C,d_conv)
        out = (window * w).sum(-1)
        if self.conv1d.bias is not None:
            out = out + self.conv1d.bias
        return out, window[..., 1:]

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, device=None, **kwargs):
        dtype = dtype or self.out_proj.weight.dtype
        device = device or self.out_proj.weight.device
        conv_state = torch.zeros(batch_size, 3 * self.d_inner, self.d_conv - 1, device=device, dtype=dtype)
        S = torch.zeros(batch_size, self.nheads, self.headdim, self.headdim, device=device, dtype=torch.float32)
        phi = torch.zeros(batch_size, self.nheads, device=device, dtype=torch.float32)
        return conv_state, S, phi


class _NormWeight(nn.Module):
    """Holds the gated-readout RMSNorm weight (state-dict key `norm.weight`, [headdim])."""

    def __init__(self, dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
