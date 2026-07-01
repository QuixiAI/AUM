# AUM-Ø training schedule (v6 §12). Four stages, gated on a SCALE-FREE prediction-head R^2.

from dataclasses import dataclass
from enum import IntEnum


class Stage(IntEnum):
    EVIDENCE_CORE = 1        # J_t=0; train A,U,M + prediction head (L_LM + L_pred + L_mu + L_S)
    FORCED_REVISION = 2      # silence forced at J=K on a sparse subset; add L_pressure + L_consistency
    SOFT_HALTING = 3         # loss-mixture halting with p_{Jmax}=1; add lambda_C E[J]; anneal p_explore
    EVENT_TRIGGERED = 4      # hard / pressure-triggered halting; silence only at high expected benefit


@dataclass
class ScheduleConfig:
    # §12 scale-free pressure gate: enable L_pressure only once the prediction head beats the
    # trivial predictor by a margin on held-out data — 1 - ||e||^2/||g - g_bar||^2 > eta_r2.
    # Before that, b_t flows through an untrained g_hat and is noise; a pressure head trained on
    # noise labels miscalibrates stickily. R^2 is scale-free across tasks and stages.
    eta_r2: float = 0.15
    forced_K: int = 2                  # fixed silent depth for stage 2 and the §11 label
    p_explore_start: float = 0.2       # forced-silence exploration (§11)
    p_explore_floor: float = 0.02      # annealed to a FLOOR, never zero — fixed-K labels persist
    stage_fractions: tuple = (0.60, 0.20, 0.15, 0.05)   # §13 token-budget split across stages


def active_loss_terms(stage: Stage):
    """Which loss terms are active in a given stage (§12)."""
    base = {"lm", "pred", "precision", "state"}
    if stage >= Stage.FORCED_REVISION:
        base |= {"pressure", "consistency"}
    if stage >= Stage.SOFT_HALTING:
        base |= {"compute"}
    return base


def pressure_gate_clear(pred_r2, schedule: ScheduleConfig):
    """True once held-out prediction R^2 clears eta_r2 — only then may L_pressure turn on (§12)."""
    return pred_r2 > schedule.eta_r2


def p_explore_at(step, total_steps, schedule: ScheduleConfig):
    """Anneal forced-silence exploration p_explore_start -> p_explore_floor (never zero, §12)."""
    if total_steps <= 0:
        return schedule.p_explore_floor
    frac = min(max(step / total_steps, 0.0), 1.0)
    return schedule.p_explore_start + frac * (schedule.p_explore_floor - schedule.p_explore_start)
