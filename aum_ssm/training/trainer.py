# AUM-Ø trainer (v6 §10-§12): stage-aware loop tying the model, the staged schedule, the
# fixed-K counterfactual benefit signal, and the seven-term loss.

import math

import torch

from aum_ssm.training.schedule import Stage, ScheduleConfig, active_loss_terms, pressure_gate_clear
from aum_ssm.training import losses as L
from aum_ssm.training.counterfactual import rollout_benefit


class AumTrainer:
    def __init__(self, model, optimizer, config, schedule: ScheduleConfig = None,
                 accelerator=None, raw=None):
        self.model = model
        self.opt = optimizer
        self.config = config
        self.schedule = schedule or ScheduleConfig()
        self.stage = Stage.EVIDENCE_CORE
        self.max_grad_norm = 1.0
        self.force_ablation = None   # set by the gate harness to pin an ablation across all stages
        # Optional HF Accelerate integration (train/train.py): backward/clip go through the
        # accelerator (mixed precision, gradient accumulation); step/zero_grad are the prepared
        # optimizer's, which no-op on non-sync accumulation steps.
        self.accelerator = accelerator
        # Under DDP, `model` is the WRAPPED module — forward must go through it for gradient
        # sync — while attribute access (.backbone, .lm_head) needs the unwrapped module.
        self.raw = raw if raw is not None else model
        # Param split for per-group grad-norm attribution (silence block vs evidence core).
        silence = getattr(getattr(self.raw, "backbone", None), "silence", None)
        self._silence_pids = {id(p) for p in silence.parameters()} if silence is not None \
            else set()

    def _ablation_for_stage(self):
        # Stage 1 trains A,U,M + the prediction head with J=0 -> every candidate output is sigma^0's.
        return "no_op" if self.stage == Stage.EVIDENCE_CORE else None

    def train_step(self, input_ids, p_explore=None):
        cfg, terms = self.config, active_loss_terms(self.stage)
        use_silence = cfg.silence_enabled                   # False -> evidence-core baseline (LM only)
        self.raw.backbone.silence_enabled = use_silence
        ablation = self.force_ablation if self.force_ablation is not None else self._ablation_for_stage()
        parts, benefit, aux = {}, None, None

        if use_silence and "pressure" in terms:
            # Live forward (stage 2 trains AT depth K; stage 3+ trains the on-policy mixture) +
            # the policy-independent fixed-K label branches (§11), all in one call.
            train_depth = self.schedule.forced_K if self.stage == Stage.FORCED_REVISION else None
            b, y, aux, result = rollout_benefit(self.model, input_ids, cfg.beta, ablation,
                                                K=self.schedule.forced_K,
                                                train_forced_depth=train_depth, raw=self.raw)
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
            if ablation == "no_op":
                # All candidates are IDENTICAL under no_op (revision discarded), so the
                # w-weighted mixture equals plain CE on candidate 0 — same loss, same grads
                # (the w-path cancels analytically either way) at 1/3 the CE rows.
                parts["lm"] = L.lm_mixture_loss(aux.o_stack[:, :-1, :1],
                                                torch.ones_like(aux.w[:, :-1, :1]),
                                                self.raw.lm_head, input_ids[:, 1:])
            else:
                parts["lm"] = L.lm_mixture_loss(aux.o_stack[:, :-1], aux.w[:, :-1],
                                                self.raw.lm_head, input_ids[:, 1:])
            if "pred" in terms:
                parts["pred"] = L.prediction_loss(aux.g_hat, aux.g, cfg.lambda_pred)
            # Regularizers only contribute when their lambda > 0 (avoids a 0*inf from the proxies).
            if "precision" in terms and cfg.lambda_precision > 0:
                mus = [l._last_mu for l in self.raw.backbone.layers if l._last_mu is not None]
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
        # backward -> clip -> step -> zero_grad: zeroing AFTER the step keeps this correct under
        # gradient accumulation (zeroing first would wipe the accumulated grads on the sync step).
        # clip_grad_norm_ returns the PRE-clip total norm — logged as the numerical-stability
        # signal (spikes = instability; the 1.0 clip caps what the optimizer actually applies).
        grad_norm = gn_silence = gn_evidence = None
        if self.accelerator is not None:
            self.accelerator.backward(total)
            if self.accelerator.sync_gradients:
                gn_silence, gn_evidence = self._group_grad_norms()   # PRE-clip attribution
                grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(),
                                                             self.max_grad_norm)
        else:
            total.backward()
            gn_silence, gn_evidence = self._group_grad_norms()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.max_grad_norm)
        self.opt.step()
        self.opt.zero_grad()
        metrics = {"loss": float(total.detach()), "parts": used, "stage": int(self.stage)}
        if grad_norm is not None:                # present on sync micro-steps (the logged ones)
            metrics["grad_norm"] = float(grad_norm)
            if gn_silence is not None:
                metrics["grad_norm_silence"] = gn_silence
                metrics["grad_norm_evidence"] = gn_evidence
        if aux is not None:
            # E[J] directly, NOT via lambda_C*E[J] (train/compute) — lambda_C is 0 until the
            # stage-3 ramp, and halting collapse is exactly what the ramp can cause (§12).
            metrics["expected_J"] = float(aux.expected_J.detach().mean())
            # train-side R^2 (scale-free; the GATE reads the held-out version) + the silence-
            # mechanism scalars: pressure spread and how far revision moves the register.
            g = aux.g.detach().float()
            base = (g - g.mean(dim=(0, 1), keepdim=True)).pow(2).sum().clamp_min(1e-12)
            metrics["pred_r2_train"] = float(1.0 - aux.e.detach().float().pow(2).sum() / base)
            pi = aux.pi.detach().float()
            metrics["pi_mean"] = float(pi.mean())
            metrics["pi_std"] = float(pi.std())
            metrics["dsigma"] = float((aux.sigma_star - aux.sigma_traj[0])
                                      .detach().float().norm(dim=-1).mean())
            # raw consistency energy E — the hinge (L_consistency) reads ~0 both when revision is
            # working (E non-increasing) and when E is degenerate/unwired. The hinge alone can't
            # tell those apart, so log the LEVEL (E_mean, confirms E is nonzero) and the DROP across
            # silent steps (E(sigma^0) - E(sigma^{Jmax}); positive => revision is reducing tension,
            # the actual revision-is-working signal the clamp_min(0) hinge hides).
            E = aux.E_traj.detach().float()
            metrics["E_mean"] = float(E.mean())
            metrics["E_drop"] = float((E[..., 0] - E[..., -1]).mean())
            # p0/p1 directly — E[J] is a summary that hides WHICH sigmoid saturated. The
            # halting params get no gradient in stage 1, but their INPUTS drift under LM
            # training, so the frozen sigmoids can saturate (observed: p ~ 0.06 by step 240).
            w = aux.w.detach().float()
            metrics["p0_mean"] = float(w[..., 0].mean())
            metrics["p1_mean"] = float((w[..., 1] / (1 - w[..., 0]).clamp_min(1e-6)).mean())
        if benefit is not None:
            metrics["benefit_mean"] = float(benefit.mean())
            # the §12 guard: is integration pressure tracking measured counterfactual benefit?
            pi_f = aux.pi[:, :-1].detach().float().reshape(-1)
            b_f = benefit.float().reshape(-1)
            if pi_f.numel() > 1 and float(pi_f.std()) > 0 and float(b_f.std()) > 0:
                metrics["corr_pi_b"] = float(torch.corrcoef(torch.stack([pi_f, b_f]))[0, 1])
        return metrics, aux

    @torch.no_grad()
    def _group_grad_norms(self):
        """(silence, evidence) pre-clip grad norms — attributes instability to the sequential
        sigma-chain vs the evidence core. One fused norm pass; ~ms."""
        if not self._silence_pids:
            return None, None
        sil, evi = [], []
        for p in self.model.parameters():
            if p.grad is not None:
                (sil if id(p) in self._silence_pids else evi).append(p.grad)
        norms = [torch.linalg.vector_norm(
            torch.stack(torch._foreach_norm(g)) if g else torch.zeros(1, device="cpu"))
            for g in (sil, evi)]
        return float(norms[0]), float(norms[1])

    @torch.no_grad()
    def recenter_halting(self, input_ids, z_target=0.3):
        """Unsaturate the halting head at stage-3 entry (opt-in via --recenter-halting).

        The halting params receive no gradient before stage 3, but their inputs drift under
        LM training, saturating the frozen sigmoids (observed p ~ 0.06, where the sigmoid
        gradient is ~4x smaller than at 0.5) — adaptive depth would start its learning phase
        in a hole. halt_2 has no bias (and the fused-kernel ABI has no bias slot), so the
        remedy is RESCALING halt_2.weight: z = w.h shrinks toward 0, p_j moves toward 0.5,
        and the learned feature DIRECTION is preserved. p0/p1 are measured by an eval-mode
        probe forward (forced_depth=None, so aux.w is the real cascade) and mean-reduced
        across ranks — the scale must be identical everywhere or the distributed Muon's
        all_gathered params diverge. Returns (scale, p0, p1)."""
        was_training = self.model.training
        self.model.eval()
        _, aux = self.model(input_ids, return_aux=True)
        if was_training:
            self.model.train()
        w = aux.w.float()
        p = torch.stack([w[..., 0].mean(),
                         (w[..., 1] / (1 - w[..., 0]).clamp_min(1e-6)).mean()])
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            p = self.accelerator.reduce(p, reduction="mean")
        p0, p1 = (float(v.clamp(1e-4, 1 - 1e-4)) for v in p)
        z = max(abs(math.log(v / (1 - v))) for v in (p0, p1))
        scale = 1.0 if z <= z_target else z_target / z
        if scale < 1.0:
            self.raw.backbone.silence.pressure_halt.halt_2.weight.mul_(scale)
        return scale, p0, p1

    @torch.no_grad()
    def pred_val_r2(self, input_ids, return_parts=False):
        """Held-out prediction-head R^2 = 1 - ||e||^2 / ||g - g_bar||^2 — the §12 gate criterion.

        g_bar is the held-out batch mean (the running-mean estimate at eval time).
        return_parts=True also returns (resid, base) — if R^2 misbehaves, a stuck numerator
        means the head isn't learning; a collapsing denominator means g itself is losing
        variance (representation collapse), which raw R^2 would disguise as improvement.
        """
        if not self.config.silence_enabled:
            return (float("-inf"), 0.0, 0.0) if return_parts else float("-inf")
        self.raw.backbone.silence_enabled = True
        _, aux = self.model(input_ids, return_aux=True, ablation="no_op")
        g = aux.g
        resid = float((aux.e).pow(2).sum())
        base = float((g - g.mean(dim=(0, 1), keepdim=True)).pow(2).sum().clamp_min(1e-12))
        r2 = 1.0 - resid / base
        return (r2, resid, base) if return_parts else r2

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
