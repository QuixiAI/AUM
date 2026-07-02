# AUM-Ø v6 configuration. The defaults ARE the AUM-Ø-Tiny v6 reference (AUM-Ø.md §13 +
# Appendix A): ~78M total, silence block ~1.77M, silence-ablated evidence core ~76.5M. The §4
# rotation ladder constants (B = headdim/2 blocks, omega geometric in [1e-3, 1]) are fixed in
# aum_ssm.modules.ssd_reference.ladder_freqs and registered per layer as the non-trainable
# unfold.rope_freqs buffer.
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AumConfig:
    # ---- Backbone / evidence core (AUM-Ø-Tiny v6 reference, ~78M) ----
    d_model: int = 512
    n_layer: int = 12                 # L evidence layers (token-clock recurrence)
    d_intermediate: int = 1408        # SwiGLU MLP hidden width
    vocab_size: int = 49152
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True

    # Norm / residual plumbing (inherited from the Mamba scaffolding)
    norm_epsilon: float = 1e-5
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = False       # Triton fused add+norm is a NVIDIA-only optimization
    initializer_cfg: dict = field(default_factory=dict)

    # ---- A phase: bounded local GQA grounding (§3) ----
    attn_num_heads: int = 8
    attn_num_heads_kv: int = 2
    attn_head_dim: int = 64
    attn_window: Optional[int] = 256   # §13 reference: sliding window w=256 (None -> full causal)
    attn_cfg: dict = field(default_factory=dict)

    # ---- U phase: resonant affine evidence recurrence (§4) ----
    kernel_backend: str = "auto"       # auto|reference|metal|triton
    chunk_size: int = 64
    ssm_cfg: dict = field(default_factory=dict)

    # ---- Global silence block (§5-§9) ----
    silence_enabled: bool = False     # False -> evidence-core baseline (g_t output, §14)
    baseline: Optional[str] = None    # None (reference) | "top_gru" (§14 adapter baseline)
    d_sigma: int = 128                # bottlenecked hypothesis register width (C4)
    d_mu: int = 32                    # precision / error projection width (k)
    d_phase: int = 32                 # phase embedding width Φ(φ)
    j_max: int = 2                    # forced final halt depth (§8)
    kappa: float = 0.1                # consistency register-inertia weight (§7)
    entropy_feature: bool = False     # optional H_t pressure feature — the registered §14 ablation
    silence_segment: int = 64         # C7: checkpoint the global recurrence every N tokens during
                                      # training (exact gradients, boundary states only; 0 = off)

    # ---- Loss weights (§10, §13 reference values) ----
    lambda_pred: float = 0.5          # lambda_P: prediction-head objective
    lambda_pressure: float = 1.0      # integration-pressure calibration (§11)
    lambda_compute: float = 0.0       # lambda_C: E[J_t] penalty — ramped 0 -> 5e-3 in stage 3 (§12)
    lambda_consistency: float = 0.1   # lambda_E: consistency monotonicity (active from stage 2)
    lambda_precision: float = 1e-3    # lambda_mu: PER-LAYER ||mu^l||_1 only — never the global mu (§10)
    lambda_state: float = 1e-4        # lambda_S: ||S||^2

    # ---- Pressure calibration (§11) and halting (§8, §12) ----
    beta: float = 0.02                # fixed calibrated benefit transform constant
    halt_delta: float = 0.5           # inference halting threshold delta
    p_explore: float = 0.02           # forced-exploration FLOOR (§12; anneal 0.2 -> this, never 0)
