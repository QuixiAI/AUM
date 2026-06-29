# AUM-Ø training schedule (§19, §20). Four stages, gated on prediction-head validation loss.

from dataclasses import dataclass, field
from enum import IntEnum


class Stage(IntEnum):
    EVIDENCE_CORE = 1        # J_t=0; train A,U,M + prediction head (L_LM + L_pred + L_precision + L_state)
    FORCED_REVISION = 2      # force silence on a sparse subset, K in {1,2}; add L_pressure + L_consistency
    SOFT_HALTING = 3         # enable w_j with p_{Jmax}=1; add L_compute; anneal p_explore -> 0
    EVENT_TRIGGERED = 4      # fire silence only at high pi_t (inference policy)


@dataclass
class ScheduleConfig:
    # Pressure-training gate (§20): do not enable L_pressure until held-out L_pred^val < eta.
    pred_val_gate_eta: float = 0.0     # TODO(AUM): set from the evidence-core run
    forced_K: int = 2                  # forced silent depth in stage 2
    p_explore_start: float = 0.25      # forced silence exploration (§19)
    p_explore_end: float = 0.0


def active_loss_terms(stage: Stage):
    """Which loss terms are active in a given stage (§20)."""
    base = {"lm", "pred", "precision", "state"}
    if stage >= Stage.FORCED_REVISION:
        base |= {"pressure", "consistency"}
    if stage >= Stage.SOFT_HALTING:
        base |= {"compute"}
    return base


def pressure_gate_clear(pred_val_loss, schedule: ScheduleConfig):
    """True once the evidence-core prediction head is good enough to trust b_t (§20)."""
    return pred_val_loss < schedule.pred_val_gate_eta
