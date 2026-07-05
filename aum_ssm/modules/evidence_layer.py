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
        self._last_mu = None                              # per-layer mu^l stash for the §10 l1 term
        self.input_layernorm = norm_cls(dim)              # shared LN(x) for A / U / controller (§3)
        self.ground_attn = attn_cls(dim)                  # A phase
        self.unfold = unfold_cls(dim)                     # U phase
        self.modulate = modulate_cls(dim)                 # M phase
        self.post_attention_layernorm = norm_cls(dim)
        self.mlp = mlp_cls(dim)

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None,
                inference_params=None, return_silence_ctx=False, **kwargs):
        attn_cache = unfold_cache = None
        if inference_params is not None:                  # lazily allocate the layer cache if absent
            kv = inference_params.key_value_memory_dict
            if self.layer_idx not in kv:
                kv[self.layer_idx] = self.allocate_inference_cache(
                    hidden_states.shape[0], inference_params.max_seqlen)
            attn_cache, unfold_cache = kv[self.layer_idx]["attn"], kv[self.layer_idx]["unfold"]

        # fold the previous block's output into the residual stream
        residual = (hidden_states + residual) if residual is not None else hidden_states
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        compute_dtype = hidden_states.dtype
        x_bar = self.input_layernorm(residual).to(compute_dtype)
        h_A = self.ground_attn(x_bar, inference_params=inference_params, cache=attn_cache)
        if return_silence_ctx:                            # top layer: expose the read source + phase
            h_U, m_t, s_t, phi, read_src = self.unfold(
                x_bar, inference_params=inference_params, cache=unfold_cache, return_read=True)
        else:
            h_U, m_t, s_t = self.unfold(x_bar, inference_params=inference_params, cache=unfold_cache)
        h_M, _mu = self.modulate(h_A, h_U, m_t)           # h^A + h^U + Δh (§3)
        self._last_mu = _mu                               # exposed for the per-layer-only l1 (§10)

        residual = residual + h_M
        x2 = self.post_attention_layernorm(residual).to(compute_dtype)
        hidden_states = self.mlp(x2)
        if return_silence_ctx:
            # "read" is a per-token closure when decoding, or the write-tensor pack (alpha, x,
            # k_rot, ...) for the backbone's sequential global recurrence when training/prefilling.
            return hidden_states, residual, {"phi": phi, "m_t": m_t, "s_t": s_t, "read": read_src}
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {"attn": self.ground_attn.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs),
                "unfold": self.unfold.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)}
