# AUM-Ø global silence block (v6 §5-§9). The entire net-new subsystem — plain PyTorch, no kernel.
#
# Submodule shapes mirror Appendix A of AUM-Ø.md exactly (model.silence.*), so the state_dict
# layout is correct (~1.77M params). forward() runs the per-token silence map over a FIXED
# evidence state S_t, accessed only through an injected `evidence_read` callable
#   evidence_read(query_512, phi, exclude_current, pooled) -> r_512
# so this module is independent of the U-phase head geometry and unit-testable with a stub read.
#
# v6 (§8): halting mixes LOSSES, never states. Each sigma^j yields its own output o^(j)
# (aux.o_stack); training minimizes the w-weighted mixture of per-candidate LM losses; exactly one
# candidate sigma^{j*} (aux.sigma_star) is carried to t+1. The cross-token recurrence
# (sigma_prev = previous sigma_star) is the caller's job — this module is the per-token map.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _phase_embed(phi, d_phase):
    """Sinusoidal embedding of the per-head resonance phase phi (..., H) -> (..., d_phase)."""
    H = phi.shape[-1]
    k = max(d_phase // (2 * H), 1)
    freqs = torch.arange(1, k + 1, device=phi.device, dtype=phi.dtype)
    ang = phi.unsqueeze(-1) * freqs                                  # (..., H, k)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)       # (..., H, 2k)
    emb = emb.reshape(*phi.shape[:-1], H * 2 * k)
    if emb.shape[-1] < d_phase:                                     # pad if not divisible
        emb = F.pad(emb, (0, d_phase - emb.shape[-1]))
    return emb[..., :d_phase]


@dataclass
class SilenceAux:
    g: torch.Tensor         # g_t (the grounded summary the block conditioned on)
    g_hat: torch.Tensor
    e: torch.Tensor
    mu: torch.Tensor
    e_tilde: torch.Tensor
    sigma0: torch.Tensor
    sigma_traj: list        # [sigma^0 .. sigma^Jmax]
    r_traj: list            # reads r(sigma^j), j=0..Jmax
    E_traj: torch.Tensor    # (B, L, Jmax+1)
    pi: torch.Tensor        # (B, L)
    w: torch.Tensor         # (B, L, Jmax+1) halting weights
    expected_J: torch.Tensor  # (B, L)
    o_stack: torch.Tensor   # (B, L, Jmax+1, d) per-candidate outputs — the §8 loss mixture
    j_star: torch.Tensor    # (B, L) long — the single carried candidate index
    sigma_star: torch.Tensor  # (B, L, d_sigma) — sigma^{j*}, the ONLY register carried to t+1
    phi: torch.Tensor       # (B, L, H) the per-head phase — for the §14 phase-distance falsifier


class PredictiveGrounding(nn.Module):
    """Hypothesis-conditioned predictive grounding: predict g_t through the previous hypothesis (§4)."""

    def __init__(self, d_model, d_sigma, d_phase, device=None, dtype=None):
        kw = {"device": device, "dtype": dtype}
        super().__init__()
        self.query_proj = nn.Linear(d_sigma, d_model, bias=False, **kw)   # W_q^pred: sigma -> state-key space
        self.read_proj = nn.Linear(d_model, d_model, bias=False, **kw)    # W_R: r^pred -> d
        self.hyp_proj = nn.Linear(d_sigma, d_model, bias=False, **kw)     # W_sigma in predict
        self.phase_proj = nn.Linear(d_phase, d_model, bias=False, **kw)   # W_phi: Phi(phi) -> d
        self.out_proj = nn.Linear(d_model, d_model, bias=False, **kw)     # W_P
        self.norm = nn.LayerNorm(d_model, **kw)


class GlobalModulate(nn.Module):
    """Error-fed precision (precision only, no readout adapter) (§8)."""

    def __init__(self, d_model, d_sigma, d_mu, device=None, dtype=None):
        kw = {"device": device, "dtype": dtype}
        super().__init__()
        self.err_proj = nn.Linear(d_model, d_mu, bias=False, **kw)        # W_e
        self.in_proj_mu = nn.Linear(2 * d_model + d_sigma + d_mu, d_mu, bias=False, **kw)  # W_mu


class Register(nn.Module):
    """Bottlenecked hypothesis register: init + nonlinear gated revision loop (§9, §10)."""

    def __init__(self, d_model, d_sigma, d_mu, device=None, dtype=None):
        kw = {"device": device, "dtype": dtype}
        super().__init__()
        self.init_proj = nn.Linear(d_sigma + d_model + 2 * d_mu, d_sigma, bias=False, **kw)  # W_sigma0
        self.read_proj = nn.Linear(d_sigma, d_model, bias=False, **kw)    # W_q^sigma
        upd_in = d_sigma + d_model + 2 * d_mu + d_model
        self.update_gate = nn.Linear(upd_in, d_sigma, bias=False, **kw)   # W_g
        self.update_cand = nn.Linear(upd_in, d_sigma, bias=False, **kw)   # W_n
        self.norm = nn.LayerNorm(d_sigma, **kw)                           # shared by init (LN) + revise


class Consistency(nn.Module):
    """Precision-weighted consistency functional E_t(sigma) (§11)."""

    def __init__(self, d_model, d_sigma, d_mu, device=None, dtype=None):
        kw = {"device": device, "dtype": dtype}
        super().__init__()
        self.P_G = nn.Linear(d_model, d_sigma, bias=False, **kw)
        self.Q_G = nn.Linear(d_sigma, d_sigma, bias=False, **kw)
        self.P_R = nn.Linear(d_model, d_sigma, bias=False, **kw)
        self.Q_R = nn.Linear(d_sigma, d_sigma, bias=False, **kw)
        self.prec_G = nn.Linear(d_mu, d_sigma, bias=False, **kw)
        self.prec_R = nn.Linear(d_mu, d_sigma, bias=False, **kw)


class PressureHalt(nn.Module):
    """Integration pressure pi_t (§9) and the halting policy (§8).

    v6: full-vocabulary predictive entropy H_t is NOT a base feature (it costs a 49k softmax on
    the critical path); `entropy_feature=True` re-adds it for the registered §14 ablation only.
    """

    def __init__(self, d_model, d_sigma, halt_hidden=64, entropy_feature=False,
                 device=None, dtype=None):
        kw = {"device": device, "dtype": dtype}
        super().__init__()
        n_scalars = 3 + int(entropy_feature)                                   # [Δe, ΔσR, s] (+H)
        self.pressure_in = nn.Linear(d_model + n_scalars, d_sigma, bias=False, **kw)
        self.pressure_out = nn.Parameter(torch.zeros(d_sigma, **kw))           # w_pi (vector)
        self.halt_1 = nn.Linear(d_sigma + 2, halt_hidden, bias=False, **kw)    # [sigma, pi, E]
        self.halt_2 = nn.Linear(halt_hidden, 1, bias=False, **kw)


class SilenceBlock(nn.Module):
    """The single global silence block on top of the evidence stack (§3)."""

    def __init__(self, d_model, d_sigma=128, d_mu=32, d_phase=32, j_max=2, kappa=0.1,
                 halt_delta=0.5, pi_trigger=None, entropy_feature=False, top_gru=False,
                 device=None, dtype=None):
        kw = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model, self.d_sigma, self.d_mu, self.d_phase = d_model, d_sigma, d_mu, d_phase
        self.j_max = j_max
        self.kappa = kappa
        self.halt_delta = halt_delta          # inference halting threshold delta (§8)
        self.pi_trigger = pi_trigger          # stage-4 policy J(pi): depth j_max iff pi clears this
        self.entropy_feature = entropy_feature
        self.top_gru = top_gru      # Top-GRU adapter baseline (§14): generic recurrent, no S read

        self.predict = PredictiveGrounding(d_model, d_sigma, d_phase, **kw)
        self.modulate = GlobalModulate(d_model, d_sigma, d_mu, **kw)
        self.register = Register(d_model, d_sigma, d_mu, **kw)
        self.consistency = Consistency(d_model, d_sigma, d_mu, **kw)
        self.pressure_halt = PressureHalt(d_model, d_sigma, entropy_feature=entropy_feature, **kw)

        self.condition_out = nn.Linear(d_model, d_model, bias=False, **kw)        # W_o
        self.condition_out_sigma = nn.Linear(d_sigma, d_model, bias=False, **kw)  # W_sigma
        self.condition_norm = nn.LayerNorm(d_model, **kw)

        if top_gru:  # baseline: revise sigma with a GRU on [g, e_tilde, mu] and no S read,
            #          but KEEP a pooled-evidence prediction head (§14) so e_t is not handicapped
            #          and the single ablated factor is the silent read of S through sigma.
            self.gru = nn.GRUCell(d_model + 2 * d_mu, d_sigma, **kw)
            self.pool_proj = nn.Linear(d_model, d_model, bias=False, **kw)        # W_S
            # Pool(S) = S q_pool with a LEARNED static per-head query — phase-free and sigma-free,
            # the strongest evidence summary available without the ablated addressing (C9: the
            # baseline is generous to the null). Mean pooling is its uniform-query special case.
            self.pool_query = nn.Parameter(torch.randn(d_model, **kw) / d_model ** 0.5)

    # ---- §5-§9 helpers (batch-agnostic; operate over any leading (B,L) dims) ----
    def _predict(self, g_t, sigma_prev, phi_prev, evidence_read, no_read):
        if self.top_gru:
            # §14 Top-GRU baseline: pooled-evidence prediction head — Pool(S_{t-1}) = S q_pool
            # replaces the sigma-conditioned read, so e_t is not handicapped and the ablated
            # factor is precisely the silent read of S through sigma.
            q = self.pool_query.expand(*sigma_prev.shape[:-1], -1)
            pooled = evidence_read(q, phi_prev, exclude_current=True, pooled=True)
            pre = (self.pool_proj(pooled) + self.predict.hyp_proj(sigma_prev)
                   + self.predict.phase_proj(_phase_embed(phi_prev, self.d_phase)))
        else:
            if getattr(self, "_zero_sigma_in_predict", False):
                # §16 sigma-relevance check: zero the sigma input to the prediction head and
                # measure g_hat degradation — detects the silent W_q^pred -> 0 bypass failure.
                sigma_prev = torch.zeros_like(sigma_prev)
            # The no-read control (§14) zeroes the SILENT read r^j only; the predictive read is
            # part of the prediction head (C5) and stays intact, so the control isolates r^j.
            q_pred = self.predict.query_proj(sigma_prev)
            r_pred = evidence_read(q_pred, phi_prev, exclude_current=True)
            pre = (self.predict.read_proj(r_pred) + self.predict.hyp_proj(sigma_prev)
                   + self.predict.phase_proj(_phase_embed(phi_prev, self.d_phase)))
        g_hat = self.predict.out_proj(self.predict.norm(pre))
        return g_hat, g_t - g_hat

    def _precision(self, g_t, e, sigma_prev, m_t):
        mu = torch.sigmoid(self.modulate.in_proj_mu(torch.cat([g_t, e, sigma_prev, m_t], -1)))
        e_tilde = mu * self.modulate.err_proj(e)
        return mu, e_tilde

    def _consistency(self, g_t, sigma, r_sigma, mu, sigma_prev):
        # §7: precision enters DETACHED — it weights the diagnostic, but gradient descent cannot
        # shrink E by turning precision off (the v5.3 anti-revision escape hatch, welded shut).
        mu = mu.detach()
        dG = self.consistency.P_G(g_t) - self.consistency.Q_G(sigma)
        dR = self.consistency.P_R(r_sigma) - self.consistency.Q_R(sigma)
        muG, muR = self.consistency.prec_G(mu), self.consistency.prec_R(mu)
        return ((muG * dG).pow(2).sum(-1) + (muR * dR).pow(2).sum(-1)
                + self.kappa * (sigma - sigma_prev).pow(2).sum(-1))

    def forward(self, g_t, evidence_read, phi_t, phi_prev=None, sigma_prev=None,
                m_t=None, s_t=None, logits_fn=None, ablation: Optional[str] = None,
                forced_depth: Optional[int] = None, halt_u: Optional[torch.Tensor] = None):
        """Revise the hypothesis register and emit the conditioned output o_t (§5-§9).

        g_t: (B,L,d_model). evidence_read(query_512, phi, exclude_current, pooled) -> (B,L,d_model).
        phi_t/phi_prev: (B,L,H). sigma_prev: (B,L,d_sigma). m_t: (B,L,d_mu). s_t: (B,L,1).
        forced_depth=K forces the halting distribution to one-hot at K (§11 fixed-depth label /
        §12 stage-2 forced revision).

        Returns (o_t, SilenceAux) with o_t = o^{(j*)}. §8: halting mixes LOSSES, never states —
        aux.o_stack (B,L,Jmax+1,d) + aux.w carry the training mixture; aux.sigma_star is the
        single coherent candidate the caller carries into t+1 (j* ~ Categorical(w) in training,
        j* = min{j: p_j >= delta} at inference). No convex blend of hypotheses ever exists.
        Cross-token sigma carry (sigma_prev = prev sigma_star) is the caller's job.
        """
        *lead, _ = g_t.shape
        if phi_prev is None:
            phi_prev = phi_t
        if sigma_prev is None:
            sigma_prev = g_t.new_zeros(*lead, self.d_sigma)
        if m_t is None:
            m_t = g_t.new_zeros(*lead, self.d_mu)
        if s_t is None:
            s_t = g_t.new_zeros(*lead, 1)
        no_read = ablation == "no_read"
        if ablation == "phase_scrambled" and phi_t.shape[-2] > 1:
            # §14 control: q = R(phi_t + eps_t) with eps shuffled ACROSS TOKENS — read at another
            # token's phase. Only the silent read uses phi_t here; predict uses phi_prev.
            perm = torch.randperm(phi_t.shape[-2], device=phi_t.device)
            phi_t = phi_t[..., perm, :]

        def read(sigma):
            r = evidence_read(self.register.read_proj(sigma), phi_t, exclude_current=False)
            return torch.zeros_like(r) if no_read else r

        # §5 predictive grounding, §6 precision, §7 register init
        g_hat, e = self._predict(g_t, sigma_prev, phi_prev, evidence_read, no_read)
        mu, e_tilde = self._precision(g_t, e, sigma_prev, m_t)
        sigma0 = self.register.norm(self.register.init_proj(
            torch.cat([sigma_prev, g_t, e_tilde, mu], -1)))

        # §8 revision loop
        sigma_traj, r_traj = [sigma0], []
        sigma_j = sigma0
        for _ in range(self.j_max):
            if self.top_gru:                                    # generic recurrent adapter, no S read
                r_j = torch.zeros_like(g_t)
                inp = torch.cat([g_t, e_tilde, mu], -1)
                sigma_j = self.gru(inp.reshape(-1, inp.shape[-1]),
                                   sigma_j.reshape(-1, self.d_sigma)).reshape(sigma_j.shape)
            else:
                r_j = read(sigma_j)
                z = torch.cat([sigma_j, g_t, e_tilde, mu, r_j], -1)
                sigma_j = self.register.norm(
                    sigma_j + torch.sigmoid(self.register.update_gate(z))
                    * torch.tanh(self.register.update_cand(z)))
            r_traj.append(r_j)
            sigma_traj.append(sigma_j)
        r_traj.append(torch.zeros_like(g_t) if self.top_gru else read(sigma_j))

        # §7 consistency over the trajectory (detached-mu weighting inside)
        E_traj = torch.stack([self._consistency(g_t, sigma_traj[j], r_traj[j], mu, sigma_prev)
                              for j in range(self.j_max + 1)], dim=-1)  # (B,L,Jmax+1)

        # §9 integration pressure — cheap features; H_t only behind the registered ablation flag
        delta_e = e_tilde.norm(dim=-1)
        delta_sR = (self.consistency.P_R(r_traj[0]) - self.consistency.Q_R(sigma0)).norm(dim=-1)
        zeta = g_t                                              # Pool_zeta = identity (no Appendix params)
        feats = [zeta]
        if self.entropy_feature:
            if logits_fn is not None:
                p0 = torch.softmax(logits_fn(self._output(g_t, sigma0)), dim=-1)
                H_t = -(p0 * torch.log(p0.clamp_min(1e-9))).sum(-1)
            else:
                H_t = g_t.new_zeros(*lead)
            feats.append(H_t.unsqueeze(-1))
        feats += [delta_e.unsqueeze(-1), delta_sR.unsqueeze(-1), s_t]
        pi = F.softplus((self.pressure_halt.pressure_out
                         * torch.tanh(self.pressure_halt.pressure_in(torch.cat(feats, -1)))).sum(-1))

        # §8 halting probabilities (forced p_{Jmax}=1)
        ps = []
        for j in range(self.j_max + 1):
            if j < self.j_max:
                hin = torch.cat([sigma_traj[j], pi.unsqueeze(-1), E_traj[..., j:j + 1]], -1)
                ps.append(torch.sigmoid(self.pressure_halt.halt_2(
                    torch.tanh(self.pressure_halt.halt_1(hin)))).squeeze(-1))
            else:
                ps.append(torch.ones_like(pi))
        w, cum = [], torch.ones_like(pi)
        for j in range(self.j_max + 1):
            w.append(ps[j] * cum)
            cum = cum * (1 - ps[j])
        w = torch.stack(w, dim=-1)                              # (B,L,Jmax+1), sums to 1

        # §8 loss-mixture halting: per-candidate outputs; carry exactly ONE candidate.
        sigma_stack = torch.stack(sigma_traj, dim=-2)           # (B,L,Jmax+1,d_sigma)
        if ablation == "no_op":                                 # loop ran; revision discarded (§14)
            sigma_stack = sigma_traj[0].unsqueeze(-2).expand_as(sigma_stack).contiguous()
        o_stack = self._output(g_t.unsqueeze(-2), sigma_stack)  # (B,L,Jmax+1,d_model)

        if forced_depth is not None:                            # fixed-depth K (§11 label / stage 2)
            j_star = torch.full(lead, int(forced_depth), device=w.device, dtype=torch.long)
            w = F.one_hot(j_star, self.j_max + 1).to(w.dtype)
        elif self.training:                                     # j* ~ Categorical(w); gradient flows
            if halt_u is not None:                              # through the loss mixture, not j*.
                # externally-drawn uniform -> inverse-CDF sample: deterministic given halt_u, so
                # gradient-checkpoint recompute reproduces the same j* exactly.
                hit = (w.cumsum(-1) >= halt_u.unsqueeze(-1)).to(torch.uint8)
                j_star = hit.argmax(dim=-1)
            else:
                j_star = torch.multinomial(
                    w.reshape(-1, self.j_max + 1), 1).reshape(lead)
        elif self.pi_trigger is not None:                       # §8/§12 stage-4 policy J(pi):
            fire = pi > self.pi_trigger                         # full depth only at high expected
            j_star = torch.where(fire,                          # benefit, else no revision at all
                                 torch.full(lead, self.j_max, device=w.device, dtype=torch.long),
                                 torch.zeros(lead, device=w.device, dtype=torch.long))
        else:                                                   # inference: j* = min{j: p_j >= delta}
            hit = (torch.stack(ps, dim=-1) >= self.halt_delta).to(torch.uint8)
            j_star = hit.argmax(dim=-1)                         # first j whose p clears delta
        expected_J = (w * torch.arange(self.j_max + 1, device=w.device, dtype=w.dtype)).sum(-1)

        if ablation == "random":                                # fire at random tokens, matched E[J] (§14)
            p = (expected_J / self.j_max).mean().detach().clamp(0, 1)
            fire = torch.rand(*lead, device=w.device) < p
            j_star = torch.where(fire, j_star, torch.zeros_like(j_star))

        idx = j_star[..., None, None]
        sigma_star = sigma_stack.gather(-2, idx.expand(*lead, 1, self.d_sigma)).squeeze(-2)
        o_t = o_stack.gather(-2, idx.expand(*lead, 1, self.d_model)).squeeze(-2)

        aux = SilenceAux(g_t, g_hat, e, mu, e_tilde, sigma0, sigma_traj, r_traj, E_traj, pi, w,
                         expected_J, o_stack, j_star, sigma_star, phi_t)
        return o_t, aux

    def _output(self, g_t, sigma):
        return self.condition_out(self.condition_norm(g_t + self.condition_out_sigma(sigma)))
