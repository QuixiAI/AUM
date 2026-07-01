# AUM-Ø language model: evidence core (L layers of A->U->M->MLP) + global silence block + LM head.
# Forked from mamba_ssm.models.mixer_seq_simple (Mamba scaffolding); see AUM-Ø.md §3-§14.

import math
from functools import partial
import json
import os

from collections import namedtuple

import torch
import torch.nn as nn

from aum_ssm.models.config_aum import AumConfig
from aum_ssm.modules.unfold import Unfold                 # U phase
from aum_ssm.modules.ground_attn import GroundAttn        # A phase
from aum_ssm.modules.modulate import PrecisionModulate    # M phase
from aum_ssm.modules.mlp import GatedMLP
from aum_ssm.modules.evidence_layer import EvidenceLayer
from aum_ssm.modules.norm import RMSNorm                  # pure-PyTorch; runs on CPU/MPS/CUDA
from aum_ssm.modules.silence import SilenceBlock
from aum_ssm.utils.generation import GenerationMixin
from aum_ssm.utils.hf import load_config_hf, load_state_dict_hf

# Triton fused add+norm is a NVIDIA-only optimization (enabled via fused_add_norm); optional.
try:
    from aum_ssm.ops.triton.layer_norm import layer_norm_fn, rms_norm_fn
except ImportError:
    layer_norm_fn, rms_norm_fn = None, None


def create_evidence_layer(
    d_model,
    d_intermediate,
    chunk_size,
    kernel_backend,
    attn_num_heads,
    attn_num_heads_kv,
    attn_head_dim,
    attn_window,
    norm_epsilon=1e-5,
    rms_norm=True,
    residual_in_fp32=True,
    layer_idx=None,
    device=None,
    dtype=None,
):
    """Build one evidence-core layer: A (ground_attn) -> U (unfold) -> M (modulate) -> MLP (§4-§8)."""
    factory = {"device": device, "dtype": dtype}
    norm_cls = partial(RMSNorm if rms_norm else nn.LayerNorm, eps=norm_epsilon, **factory)
    attn_cls = partial(
        GroundAttn, num_heads=attn_num_heads, num_heads_kv=attn_num_heads_kv,
        head_dim=attn_head_dim, window_size=attn_window, layer_idx=layer_idx, **factory,
    )
    unfold_cls = partial(Unfold, chunk_size=chunk_size, kernel_backend=kernel_backend,
                         layer_idx=layer_idx, **factory)
    modulate_cls = partial(PrecisionModulate, **factory)
    mlp_cls = partial(GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory)
    block = EvidenceLayer(d_model, attn_cls, unfold_cls, modulate_cls, mlp_cls,
                          norm_cls=norm_cls, residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block


def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True,
                  n_residuals_per_layer=2):
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Scale residual-path output projections by 1/sqrt(N) (GPT-2 / Megatron scheme).
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight", "o_proj.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class AumBackbone(nn.Module):
    """Evidence core (L token-clock layers) followed by one global silence block (§3)."""

    def __init__(self, config: AumConfig, device=None, dtype=None):
        factory = {"device": device, "dtype": dtype}
        super().__init__()
        self.config = config
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm
        if self.fused_add_norm and (layer_norm_fn is None or rms_norm_fn is None):
            raise ImportError("fused_add_norm requires the Triton LayerNorm kernels (NVIDIA)")

        self.embedding = nn.Embedding(config.vocab_size, config.d_model, **factory)
        self.layers = nn.ModuleList([
            create_evidence_layer(
                config.d_model, config.d_intermediate, config.chunk_size, config.kernel_backend,
                config.attn_num_heads, config.attn_num_heads_kv, config.attn_head_dim,
                config.attn_window, norm_epsilon=config.norm_epsilon, rms_norm=config.rms_norm,
                residual_in_fp32=config.residual_in_fp32, layer_idx=i, **factory,
            )
            for i in range(config.n_layer)
        ])

        # Global silence block — single block on top of the evidence stack (§0.1, §3).
        self.silence = SilenceBlock(config.d_model, d_sigma=config.d_sigma, d_mu=config.d_mu,
                                    d_phase=config.d_phase, j_max=config.j_max, **factory)
        # TODO(AUM/Phase 2): enable once SilenceBlock.forward is implemented. While False, the
        # backbone returns g_t alone (the silence-ablated baseline, §22).
        self.silence_enabled = False

        self.norm_f = (RMSNorm if config.rms_norm else nn.LayerNorm)(
            config.d_model, eps=config.norm_epsilon, **factory
        )
        self.apply(partial(_init_weights, n_layer=config.n_layer,
                           **(config.initializer_cfg or {})))

    def forward(self, input_ids, inference_params=None, **kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual,
                                            inference_params=inference_params, **kwargs)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual)              # g_t (top-of-stack grounded summary)
        if self.silence_enabled:
            hidden_states = self.silence(hidden_states, inference_params=inference_params)
        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
                for i, layer in enumerate(self.layers)}


class AumLMHeadModel(nn.Module, GenerationMixin):
    def __init__(self, config: AumConfig, device=None, dtype=None):
        super().__init__()
        self.config = config
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple)
        factory = {"device": device, "dtype": dtype}
        self.backbone = AumBackbone(config, **factory)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False, **factory)
        self.apply(partial(_init_weights, n_layer=config.n_layer, **(config.initializer_cfg or {})))
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **kwargs):
        hidden_states = self.backbone(input_ids, inference_params=inference_params, **kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config = AumConfig(**load_config_hf(pretrained_model_name))
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=4)
