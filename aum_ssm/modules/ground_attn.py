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

    def forward(self, x, inference_params=None, **kwargs):
        if inference_params is not None:
            raise NotImplementedError("GroundAttn decode path is a later milestone (Phase 3)")
        B, L, _ = x.shape
        q = rearrange(self.q_proj(x), "b l (h d) -> b l h d", d=self.head_dim)
        k = rearrange(self.k_proj(x), "b l (h d) -> b l h d", d=self.head_dim)
        v = rearrange(self.v_proj(x), "b l (h d) -> b l h d", d=self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # GQA expand
        rep = self.num_heads // self.num_heads_kv
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
        q, k, v = (rearrange(t, "b l h d -> b h l d") for t in (q, k, v))

        if self.window_size is not None and self.causal:
            attn_mask = self._window_mask(L, x.device)
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            ctx = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        ctx = rearrange(ctx, "b h l d -> b l (h d)")
        return self.o_proj(ctx)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError("GroundAttn decode cache is a later milestone (Phase 3)")
