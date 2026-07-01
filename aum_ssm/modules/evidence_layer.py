# Copyright (c) 2026.
# AUM-Ø evidence-core layer (§4-§8): A (ground_attn) -> U (unfold) -> M (modulate) -> MLP,
# with h^M accumulated into the residual stream. One shared LN(x) feeds A, U, and the U
# controller (§5). Prenorm residual convention, matching the Mamba backbone loop:
# each layer folds the previous block output into `residual`, then returns (mlp_out, residual).

from typing import Optional

import torch
from torch import nn, Tensor


class EvidenceLayer(nn.Module):
    def __init__(
        self,
        dim,
        attn_cls,
        unfold_cls,
        modulate_cls,
        mlp_cls,
        norm_cls=nn.LayerNorm,
        residual_in_fp32=True,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.input_layernorm = norm_cls(dim)              # shared LN(x) for A / U / controller (§5)
        self.ground_attn = attn_cls(dim)                  # A phase
        self.unfold = unfold_cls(dim)                     # U phase
        self.modulate = modulate_cls(dim)                 # M phase
        self.post_attention_layernorm = norm_cls(dim)
        self.mlp = mlp_cls(dim)

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None,
                inference_params=None, return_silence_ctx=False, **kwargs):
        # fold the previous block's output into the residual stream
        residual = (hidden_states + residual) if residual is not None else hidden_states
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        x_bar = self.input_layernorm(residual)
        h_A = self.ground_attn(x_bar, inference_params=inference_params)
        if return_silence_ctx:                            # top layer: expose read closure + phase
            h_U, m_t, s_t, phi, read_fn = self.unfold(
                x_bar, inference_params=inference_params, return_read=True)
        else:
            h_U, m_t, s_t = self.unfold(x_bar, inference_params=inference_params)
        h_M, _mu = self.modulate(h_A, h_U, m_t)           # h^A + h^U + Δh (§8)

        residual = residual + h_M
        x2 = self.post_attention_layernorm(residual)
        hidden_states = self.mlp(x2)
        if return_silence_ctx:
            return hidden_states, residual, {"phi": phi, "m_t": m_t, "s_t": s_t, "read_fn": read_fn}
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError("Evidence-layer decode cache is a later milestone (Phase 3)")
