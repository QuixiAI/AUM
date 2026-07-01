# AUM-Ø configuration. See AUM-Ø.md (Appendix A) for the reference layout.
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AumConfig:
    # ---- Backbone / evidence core (AUM-Ø-Tiny v5.3 reference, ~78M) ----
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

    # ---- A phase: bounded local GQA grounding (§4) ----
    attn_num_heads: int = 8
    attn_num_heads_kv: int = 2
    attn_head_dim: int = 64
    attn_window: Optional[int] = None  # None -> full causal; int -> sliding-window
    attn_cfg: dict = field(default_factory=dict)

    # ---- U phase: resonant affine evidence recurrence (§5-§6) ----
    kernel_backend: str = "auto"       # auto|reference|metal|triton
    chunk_size: int = 64
    ssm_cfg: dict = field(default_factory=dict)

    # ---- Global silence block (§3-§14) ----
    silence_enabled: bool = False     # False -> evidence-core baseline (g_t output, §22)
    d_sigma: int = 128                # bottlenecked hypothesis register width (§0.4)
    d_mu: int = 32                    # precision / error projection width (k)
    d_phase: int = 32                 # phase embedding width Φ(φ)
    j_max: int = 2                    # forced final halt depth (§13, §15)
    kappa: float = 0.1                # consistency register-inertia weight (§11)

    # ---- Loss weights (§18) ----
    lambda_pred: float = 1.0          # prediction-head objective (§16)
    lambda_pressure: float = 1.0      # integration-pressure calibration (§17)
    lambda_compute: float = 0.0       # E[J_t] compute penalty (enabled stage 3)
    lambda_consistency: float = 0.0   # consistency monotonicity (enabled stage 2)
    lambda_precision: float = 0.0     # ||mu||_1
    lambda_state: float = 0.0         # ||S||^2

    # ---- Pressure calibration (§17) and halting (§13) ----
    beta: float = 0.02                # fixed calibrated benefit transform constant
    halt_delta: float = 0.5           # hard-halting threshold at inference
    p_explore: float = 0.0            # forced silence exploration prob (§19), annealed to 0
