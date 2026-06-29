# AUM-Ø trainer skeleton (§17-§21). Ties the model, the staged schedule, the counterfactual
# benefit signal, and the seven-term loss into one loop. Net-new.

from aum_ssm.training.schedule import Stage, ScheduleConfig, active_loss_terms


class AumTrainer:
    """Stage-aware training loop for AUM-Ø.

    TODO(AUM): implement
      - the four-stage progression (§20) with the pressure-training gate,
      - forced silence exploration (§19),
      - per-step counterfactual benefit + calibrated target (§17),
      - the seven-term loss (§18),
      - the diagnostics in §21 (corr(pi, b), sigma-decode probe, Delta-sigma quartet).
    """

    def __init__(self, model, optimizer, config, schedule: ScheduleConfig = None):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.schedule = schedule or ScheduleConfig()
        self.stage = Stage.EVIDENCE_CORE

    def train_step(self, batch):
        raise NotImplementedError("Implement the staged AUM-Ø training step (§17-§20).")

    def maybe_advance_stage(self, metrics):
        """Advance EVIDENCE_CORE -> FORCED_REVISION -> ... per the §20 gates."""
        raise NotImplementedError("Implement stage transitions (§20).")
