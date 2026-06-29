# AUM-Ø language model: evidence core (L layers) + global silence block + LM head.
# Forked from mamba_ssm.models.mixer_seq_simple (Mamba scaffolding); see AUM-Ø.md §3, §14.

import math
from functools import partial
import json
import os
import copy

from collections import namedtuple

import torch
import torch.nn as nn

from aum_ssm.models.config_aum import AumConfig
from aum_ssm.modules.unfold import Mamba3 as Unfold          # U phase. TODO(AUM): add write-gate ρ_t + k/v L2-norm
from aum_ssm.modules.ground_attn import MHA as GroundAttn    # A phase. TODO(AUM): local window + QK-norm
from aum_ssm.modules.mlp import GatedMLP
from aum_ssm.modules.evidence_layer import Block as EvidenceLayer  # TODO(AUM): fold A->U->M into one layer
from aum_ssm.modules.silence import SilenceBlock
from aum_ssm.utils.generation import GenerationMixin
from aum_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from aum_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def create_evidence_layer(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    device=None,
    dtype=None,
):
    """Build one evidence-core layer.

    TODO(AUM): a full evidence layer runs A (ground_attn) -> U (unfold) -> M (modulate) -> MLP,
    accumulating h^M into the residual stream (§4-§8). This skeleton wires only the U-phase mixer +
    MLP so the backbone is structurally present and importable; A and M are folded in later.
    """
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    ssm_cfg = copy.deepcopy(ssm_cfg)
    mixer_cls = partial(Unfold, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    mlp_cls = partial(GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs)
    block = EvidenceLayer(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class AumBackbone(nn.Module):
    """Evidence core (L token-clock layers) followed by one global silence block (§3)."""

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm: bool = True,
        residual_in_fp32: bool = True,
        d_sigma: int = 128,
        j_max: int = 2,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_evidence_layer(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        # Global silence block — single block on top of the evidence stack (§0.1, §3).
        self.silence = SilenceBlock(d_model, d_sigma=d_sigma, j_max=j_max, **factory_kwargs)
        # TODO(AUM): enable once SilenceBlock.forward is implemented. While False, the backbone
        # produces logits from the grounded summary g_t alone (the silence-ablated path, §22 baseline).
        self.silence_enabled = False

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )
        # hidden_states is the grounded summary g_t (top of stack).
        if self.silence_enabled:
            # TODO(AUM): revise hypothesis register and fold sigma_bar into the output (§9-§14):
            #   o_t = W_o LN(g_t + W_sigma sigma_bar)
            hidden_states = self.silence(hidden_states, inference_params=inference_params)
        return hidden_states


class AumLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        config: AumConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = AumBackbone(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_cfg=attn_cfg,
            norm_epsilon=config.norm_epsilon,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            d_sigma=config.d_sigma,
            j_max=config.j_max,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = AumConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """Minimal save_pretrained: model state_dict + config.json."""
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=4)
