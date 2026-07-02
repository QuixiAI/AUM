# Copyright (c) 2026.
# AUM-Ø A phase: bounded local GQA grounding (§4), Appendix-A layout (ground_attn.*).
# Grouped-query attention with QK-norm and optional sliding-window causal masking. No rotary
# (Appendix A lists no rotary params for ground_attn). Training-forward for now; the KV-cache
# decode path is added with the rest of inference (Phase 3 / M2).

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from aum_ssm.modules.norm import RMSNorm


class GroundAttn(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads=8,
        num_heads_kv=2,
        head_dim=64,
        window_size=None,     # None -> full causal; int -> sliding-window causal (bounded grounding)
        qk_norm=True,
        causal=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory = {"device": device, "dtype": dtype}
        super().__init__()
        assert num_heads % num_heads_kv == 0
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.head_dim = head_dim
        self.window_size = window_size
        self.causal = causal
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False, **factory)
        self.k_proj = nn.Linear(d_model, num_heads_kv * head_dim, bias=False, **factory)
        self.v_proj = nn.Linear(d_model, num_heads_kv * head_dim, bias=False, **factory)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False, **factory)
        self.q_norm = RMSNorm(head_dim, **factory) if qk_norm else None
        self.k_norm = RMSNorm(head_dim, **factory) if qk_norm else None

    def _window_mask(self, L, device):
        # allow key j for query i iff  i - window < j <= i   (causal + bounded lookback)
        i = torch.arange(L, device=device)[:, None]
        j = torch.arange(L, device=device)[None, :]
        return (j <= i) & (j > i - self.window_size)

    def _sliding_blocks(self, q, k, v):
        """Exact sliding-window causal attention in O(L*w) memory (w = window_size).

        SDPA with a full (L, L) boolean window mask hits the MPS math fallback, which saves the
        ENTIRE quadratic attention matrix for backward — (B, H, 4096, 4096) fp32 is ~26GB per 4
        sequences, the dominant memory (and swap) cost of a train step. Instead: view the
        sequence as w-sized blocks; queries in block i attend keys in blocks {i-1, i} under a
        constant (w, 2w) relative mask — identical math, O(L*w) footprint. Checkpointed under
        training so backward recomputes the block scores instead of storing them.
        """
        w = self.window_size
        B, H, L, dh = q.shape
        nb = (L + w - 1) // w
        pad = nb * w - L
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))

        def blocks(qp, kp, vp):
            qb = qp.reshape(B, H, nb, w, dh)
            kb = kp.reshape(B, H, nb, w, dh)
            vb = vp.reshape(B, H, nb, w, dh)
            k2 = torch.cat([torch.cat([torch.zeros_like(kb[:, :, :1]), kb[:, :, :-1]], dim=2),
                            kb], dim=3)                        # (B, H, nb, 2w, dh)
            v2 = torch.cat([torch.cat([torch.zeros_like(vb[:, :, :1]), vb[:, :, :-1]], dim=2),
                            vb], dim=3)
            p = torch.arange(w, device=qp.device)[:, None]
            c = torch.arange(2 * w, device=qp.device)[None, :]
            allow = (c > p) & (c <= p + w)                     # key offset (c - w) in (p - w, p]
            allow = allow.expand(nb, w, 2 * w).clone()
            allow[0] &= (c >= w)                               # block 0: no previous block
            bias = torch.where(allow, 0.0, float("-inf")).to(qp.dtype)
            scores = qb @ k2.transpose(-1, -2) * (dh ** -0.5) + bias
            out = torch.softmax(scores, dim=-1) @ v2           # (B, H, nb, w, dh)
            return out.reshape(B, H, nb * w, dh)

        if self.training and torch.is_grad_enabled():
            ctx = torch.utils.checkpoint.checkpoint(blocks, q, k, v, use_reentrant=False)
        else:
            ctx = blocks(q, k, v)
        return ctx[:, :, :L]

    def _cache_mask(self, Lq, Lk, device):
        # query row i sits at absolute position (Lk - Lq + i); attend key j <= pos (+ window)
        pos = torch.arange(Lq, device=device)[:, None] + (Lk - Lq)
        j = torch.arange(Lk, device=device)[None, :]
        m = j <= pos
        if self.window_size is not None:
            m = m & (j > pos - self.window_size)
        return m

    def forward(self, x, inference_params=None, cache=None, **kwargs):
        B, L, _ = x.shape
        q = rearrange(self.q_proj(x), "b l (h d) -> b l h d", d=self.head_dim)
        k = rearrange(self.k_proj(x), "b l (h d) -> b l h d", d=self.head_dim)
        v = rearrange(self.v_proj(x), "b l (h d) -> b l h d", d=self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if cache is not None:                          # prefill (offset 0) or single-token decode
            k_cache, v_cache = cache
            off = inference_params.seqlen_offset
            k_cache[:, off:off + L] = k
            v_cache[:, off:off + L] = v
            k, v = k_cache[:, :off + L], v_cache[:, :off + L]
        rep = self.num_heads // self.num_heads_kv
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
        q, k, v = (rearrange(t, "b l h d -> b h l d") for t in (q, k, v))

        if cache is not None:
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=self._cache_mask(L, k.shape[2], x.device))
        elif self.window_size is not None and self.causal:
            if L > 2 * self.window_size:                  # long sequences: O(L*w) blocked form
                ctx = self._sliding_blocks(q, k, v)
            else:                                         # short: the mask is small anyway
                ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=self._window_mask(L, x.device))
        else:
            ctx = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        ctx = rearrange(ctx, "b h l d -> b l (h d)")
        return self.o_proj(ctx)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, device=None, **kwargs):
        dtype = dtype or self.o_proj.weight.dtype
        device = device or self.o_proj.weight.device
        shape = (batch_size, max_seqlen, self.num_heads_kv, self.head_dim)
        return (torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype))
