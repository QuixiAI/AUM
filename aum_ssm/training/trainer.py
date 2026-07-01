# AUM-Ø trainer (v6 §10-§12): stage-aware loop tying the model, the staged schedule, the
# fixed-K counterfactual benefit signal, and the seven-term loss.

import torch

from aum_ssm.training.schedule import Stage, ScheduleConfig, active_loss_terms, pressure_gate_clear
from aum_ssm.training import losses as L
from aum_ssm.training.counterfactual import rollout_benefit


class AumTrainer:
    def __init__(self, model, optimizer, config, schedule: ScheduleConfig = None):
        self.model = model
        self.opt = optimizer
        self.config = config
        self.schedule = schedule or ScheduleConfig()
        self.stage = Stage.EVIDENCE_CORE
        self.max_grad_norm = 1.0
        self.force_ablation = None   # set by the gate harness to pin an ablation across all stages

    def _ablation_for_stage(self):
        # Stage 1 trains A,U,M + the prediction head with J=0 -> every candidate output is sigma^0's.
        return "no_op" if self.stage == Stage.EVIDENCE_CORE else None

    def train_step(self, input_ids, p_explore=None):
        cfg, terms = self.config, active_loss_terms(self.stage)
        use_silence = cfg.silence_enabled                   # False -> evidence-core baseline (LM only)
        self.model.backbone.silence_enabled = use_silence
        ablation = self.force_ablation if self.force_ablation is not None else self._ablation_for_stage()
        parts, benefit, aux = {}, None, None

        if use_silence and "pressure" in terms:
            # Live forward (stage 2 trains AT depth K; stage 3+ trains the on-policy mixture) +
            # the policy-independent fixed-K label branches (§11), all in one call.
            train_depth = self.schedule.forced_K if self.stage == Stage.FORCED_REVISION else None
            b, y, aux, result = rollout_benefit(self.model, input_ids, cfg.beta, ablation,
                                                K=self.schedule.forced_K,
                                                train_forced_depth=train_depth)
            mask = None
            if self.stage >= Stage.SOFT_HALTING:            # §11 exploration subset (stage 2 = all
                p = self.schedule.p_explore_floor if p_explore is None else p_explore  # forced)
                mask = (torch.rand_like(b) < p).to(b.dtype)
            parts["pressure"] = cfg.lambda_pressure * L.pressure_loss(aux.pi[:, :-1], y, mask)
            logits, benefit = result.logits, b
        elif use_silence:
            result, aux = self.model(input_ids, return_aux=True, ablation=ablation)
            logits = result.logits
        else:
            logits = self.model(input_ids).logits

        if aux is not None:                                 # §8/§10 loss-mixture LM objective
            parts["lm"] = L.lm_mixture_loss(aux.o_stack[:, :-1], aux.w[:, :-1],
                                            self.model.lm_head, input_ids[:, 1:])
            if "pred" in terms:
                parts["pred"] = L.prediction_loss(aux.g_hat, aux.g, cfg.lambda_pred)
            # Regularizers only contribute when their lambda > 0 (avoids a 0*inf from the proxies).
            if "precision" in terms and cfg.lambda_precision > 0:
                mus = [l._last_mu for l in self.model.backbone.layers if l._last_mu is not None]
                if mus:                                     # PER-LAYER fields only, never global (§10)
                    parts["precision"] = L.precision_loss(mus, cfg.lambda_precision)
            if "state" in terms and cfg.lambda_state > 0:  # readout-energy proxy for ||S_t||^2
                parts["state"] = L.state_loss(torch.stack(aux.r_traj), cfg.lambda_state)
            if "consistency" in terms and cfg.lambda_consistency > 0:
                parts["consistency"] = L.consistency_loss(aux.E_traj, cfg.lambda_consistency)
            if "compute" in terms and cfg.lambda_compute > 0:
                parts["compute"] = L.compute_loss(aux.expected_J, cfg.lambda_compute)
        else:
            parts["lm"] = L.lm_loss(logits[:, :-1], input_ids[:, 1:])

        total, used = L.total_loss(parts, terms)
        self.opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.opt.step()
        metrics = {"loss": float(total.detach()), "parts": used, "stage": int(self.stage)}
        if benefit is not None:
            metrics["benefit_mean"] = float(benefit.mean())
        return metrics, aux

    @torch.no_grad()
    def pred_val_r2(self, input_ids):
        """Held-out prediction-head R^2 = 1 - ||e||^2 / ||g - g_bar||^2 — the §12 gate criterion.

        g_bar is the held-out batch mean (the running-mean estimate at eval time).
        """
        if not self.config.silence_enabled:
            return float("-inf")                        # no prediction head without silence
        self.model.backbone.silence_enabled = True
        _, aux = self.model(input_ids, return_aux=True, ablation="no_op")
        g = aux.g
        resid = (aux.e).pow(2).sum()
        base = (g - g.mean(dim=(0, 1), keepdim=True)).pow(2).sum()
        return float(1.0 - resid / base.clamp_min(1e-12))

    def maybe_advance_stage(self, pred_r2):
        """Advance stages per §12. The load-bearing gate: do NOT enable L_pressure until the
        prediction head beats the trivial predictor (else pi trains on noise labels -> sticky
        miscalibration)."""
        if self.stage == Stage.EVIDENCE_CORE and pressure_gate_clear(pred_r2, self.schedule):
            self.stage = Stage.FORCED_REVISION
        elif self.stage == Stage.FORCED_REVISION:
            self.stage = Stage.SOFT_HALTING
        elif self.stage == Stage.SOFT_HALTING:
            self.stage = Stage.EVENT_TRIGGERED
        return self.stage
