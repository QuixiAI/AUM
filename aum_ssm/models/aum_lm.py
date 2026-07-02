# AUM-Ø language model: evidence core (L layers of A->U->M->MLP) + global silence block + LM head.
# Forked from mamba_ssm.models.mixer_seq_simple (Mamba scaffolding); see AUM-Ø.md §3-§14.

import math
from functools import partial
import json
import os

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aum_ssm.models.config_aum import AumConfig
from aum_ssm.modules.unfold import Unfold                 # U phase
from aum_ssm.modules.ground_attn import GroundAttn        # A phase
from aum_ssm.modules.modulate import PrecisionModulate    # M phase
from aum_ssm.modules.mlp import GatedMLP
from aum_ssm.modules.evidence_layer import EvidenceLayer
from aum_ssm.modules.norm import RMSNorm                  # pure-PyTorch; runs on CPU/MPS/CUDA
from aum_ssm.modules.silence import SilenceBlock, SilenceAux
from aum_ssm.modules.ssd_reference import _rotate_ladder
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


def _token_read(S_prev, S_cur, headdim, freqs):
    """Per-token evidence-read closure over the sequentially-stepped top-layer state (v6 §5/§8).

    exclude_current=True -> S_{t-1} (the predictive read); else S_t (the silent read). The query is
    split per U-head and rotated by that head's ladder at the phase the CALLER passes (phi_{t-1}
    for predict, phi_t for the silent read). pooled=True -> Pool(S) = mean over the key axis (§14).
    """
    def read(query, phi_arg=None, exclude_current=False, pooled=False):
        S = S_prev if exclude_current else S_cur
        if pooled:
            return S.mean(-1).reshape(S.shape[0], 1, -1)
        qh = query.reshape(*query.shape[:-1], -1, headdim)          # (B,1,H,Dqk)
        q_rot = _rotate_ladder(qh, phi_arg, freqs)
        r = torch.einsum("bhpn,blhn->blhp", S, q_rot)
        return r.reshape(r.shape[0], r.shape[1], -1)
    return read


def _cat_aux(auxes):
    """Concatenate per-token SilenceAux (B,1,...) slices into sequence-shaped (B,L,...) aux."""
    cat = lambda xs: torch.cat(xs, dim=1)
    n_j = len(auxes[0].sigma_traj)
    return SilenceAux(
        g=cat([a.g for a in auxes]), g_hat=cat([a.g_hat for a in auxes]),
        e=cat([a.e for a in auxes]), mu=cat([a.mu for a in auxes]),
        e_tilde=cat([a.e_tilde for a in auxes]), sigma0=cat([a.sigma0 for a in auxes]),
        sigma_traj=[cat([a.sigma_traj[j] for a in auxes]) for j in range(n_j)],
        r_traj=[cat([a.r_traj[j] for a in auxes]) for j in range(len(auxes[0].r_traj))],
        E_traj=cat([a.E_traj for a in auxes]), pi=cat([a.pi for a in auxes]),
        w=cat([a.w for a in auxes]), expected_J=cat([a.expected_J for a in auxes]),
        o_stack=cat([a.o_stack for a in auxes]), j_star=cat([a.j_star for a in auxes]),
        sigma_star=cat([a.sigma_star for a in auxes]), phi=cat([a.phi for a in auxes]),
    )


class AumBackbone(nn.Module):
    """Evidence core (L token-clock layers) followed by one global silence block (§2)."""

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
                                    d_phase=config.d_phase, j_max=config.j_max, kappa=config.kappa,
                                    halt_delta=config.halt_delta,
                                    entropy_feature=getattr(config, "entropy_feature", False),
                                    top_gru=(config.baseline == "top_gru"), **factory)
        # False -> the silence-ablated evidence-core baseline (g_t output, §22).
        self.silence_enabled = config.silence_enabled

        self.norm_f = (RMSNorm if config.rms_norm else nn.LayerNorm)(
            config.d_model, eps=config.norm_epsilon, **factory
        )
        self.apply(partial(_init_weights, n_layer=config.n_layer,
                           **(config.initializer_cfg or {})))

    def forward(self, input_ids, inference_params=None, return_aux=False, ablation=None,
                forced_depth=None, **kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        ctx = None
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            top = self.silence_enabled and i == n - 1
            if top:
                hidden_states, residual, ctx = layer(
                    hidden_states, residual, inference_params=inference_params,
                    return_silence_ctx=True, **kwargs)
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params, **kwargs)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        g_t = self.norm_f(residual)                        # top-of-stack grounded summary
        if not self.silence_enabled:
            return (g_t, None) if return_aux else g_t

        phi = ctx["phi"]
        read_src, m_t, s_t = ctx["read"], ctx["m_t"], ctx["s_t"]
        logits_fn = lambda o: F.linear(o, self.embedding.weight)  # tied classifier (per-j LM mixture)
        decoding = inference_params is not None and inference_params.seqlen_offset > 0
        if decoding:               # one more step of the same recurrence: sigma_{t-1} lives in the slot
            slot = self._silence_slot(inference_params, g_t.shape[0], g_t.device)
            o_t, aux = self.silence(g_t, read_src, phi, slot["phi_prev"], slot["sigma"].unsqueeze(1),
                                    m_t, s_t, logits_fn, ablation, forced_depth)
            slot["sigma"].copy_(aux.sigma_star[:, 0])
            slot["phi_prev"].copy_(phi)
            return (o_t, aux) if return_aux else o_t

        # prefill / training: the TRUE sequential global recurrence (v6 §2/C7/§12) —
        #   sigma_{t-1} -> g_hat_t -> e_t -> mu_t -> sigma_t^0 -> (silence loop) -> sigma_t
        # run over tokens after the core's parallel scan. The loop steps the top layer's evidence
        # state itself (S_t = alpha_t*S_{t-1} + x_t (x) k_rot_t from the write pack), serving the
        # predictive read from S_{t-1}@phi_{t-1} and the silent read from S_t@phi_t. Declared
        # bottleneck (C7): sequential in T; the fused kernel / segment checkpointing comes with the
        # training bring-up.
        alpha, xw, k_rot = read_src["alpha"], read_src["x"], read_src["k_rot"]
        headdim, freqs = read_src["headdim"], read_src["freqs"]
        B, L, H, Dv = xw.shape
        S = xw.new_zeros(B, H, Dv, k_rot.shape[-1])
        sigma = g_t.new_zeros(B, 1, self.silence.d_sigma)
        phi_prev_t = phi.new_zeros(B, 1, H)
        phi_read = phi                                        # phase used by the SILENT read only
        if ablation == "phase_scrambled" and L > 1:           # §14: eps_t shuffled across tokens
            phi_read = phi[:, torch.randperm(L, device=phi.device)]
        outs, auxes = [], []
        for t in range(L):
            S_prev = S
            S = (alpha[:, t].unsqueeze(-1).unsqueeze(-1) * S_prev
                 + xw[:, t].unsqueeze(-1) * k_rot[:, t].unsqueeze(-2))
            read_t = _token_read(S_prev, S, headdim, freqs)
            o_step, aux_t = self.silence(
                g_t[:, t:t + 1], read_t, phi_read[:, t:t + 1], phi_prev_t, sigma,
                m_t[:, t:t + 1], s_t[:, t:t + 1], logits_fn,
                None if ablation == "phase_scrambled" else ablation, forced_depth)
            sigma = aux_t.sigma_star[:, :1]                   # BPTT through the token recurrence
            phi_prev_t = phi[:, t:t + 1]
            outs.append(o_step)
            auxes.append(aux_t)
        o_t = torch.cat(outs, dim=1)
        aux = _cat_aux(auxes)
        if inference_params is not None:                      # prefill: seed the silence carry slot
            slot = self._silence_slot(inference_params, g_t.shape[0], g_t.device)
            slot["sigma"].copy_(sigma[:, 0].detach())
            slot["phi_prev"].copy_(phi[:, -1:])
        return (o_t, aux) if return_aux else o_t

    def _silence_slot(self, inference_params, batch, device):
        kv = inference_params.key_value_memory_dict
        if "silence" not in kv:
            H = self.layers[-1].unfold.nheads
            kv["silence"] = {
                "sigma": torch.zeros(batch, self.silence.d_sigma, device=device, dtype=torch.float32),
                "phi_prev": torch.zeros(batch, 1, H, device=device, dtype=torch.float32),
            }
        return kv["silence"]

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        cache = {i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
                 for i, layer in enumerate(self.layers)}
        if self.silence_enabled:
            dev = self.embedding.weight.device
            H = self.layers[-1].unfold.nheads
            cache["silence"] = {
                "sigma": torch.zeros(batch_size, self.silence.d_sigma, device=dev, dtype=torch.float32),
                "phi_prev": torch.zeros(batch_size, 1, H, device=dev, dtype=torch.float32),
            }
        return cache


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

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0,
                return_aux=False, ablation=None, **kwargs):
        out = self.backbone(input_ids, inference_params=inference_params, return_aux=return_aux,
                            ablation=ablation, **kwargs)
        hidden_states, aux = out if return_aux else (out, None)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        result = CausalLMOutput(logits=lm_logits)
        return (result, aux) if return_aux else result

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
