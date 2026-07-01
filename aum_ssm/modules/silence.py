# AUM-Ø global silence block (§3-§14). The entire net-new subsystem — plain PyTorch, no kernel.
#
# Submodule shapes mirror Appendix A of AUM-Ø.md exactly (model.silence.*), so the state_dict
# layout is correct (~1.77M params). forward() runs the §4-§14 silence loop over a FIXED evidence
# state S_t, accessed only through an injected `evidence_read` callable
#   evidence_read(query_512, phi, exclude_current) -> r_512
# so this module is independent of the U-phase head geometry and unit-testable with a stub read.

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
    sigma_bar: torch.Tensor


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
    """Integration pressure pi_t (§12) and soft-halting policy (§13)."""

    def __init__(self, d_model, d_sigma, halt_hidden=64, device=None, dtype=None):
        kw = {"device": device, "dtype": dtype}
        super().__init__()
        self.pressure_in = nn.Linear(d_model + 4, d_sigma, bias=False, **kw)   # [zeta, H, Δe, ΔσR, s]
        self.pressure_out = nn.Parameter(torch.zeros(d_sigma, **kw))           # w_pi (vector)
        self.halt_1 = nn.Linear(d_sigma + 2, halt_hidden, bias=False, **kw)    # [sigma, pi, E]
        self.halt_2 = nn.Linear(halt_hidden, 1, bias=False, **kw)


class SilenceBlock(nn.Module):
    """The single global silence block on top of the evidence stack (§3)."""

    def __init__(self, d_model, d_sigma=128, d_mu=32, d_phase=32, j_max=2, kappa=0.1,
                 device=None, dtype=None):
        kw = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model, self.d_sigma, self.d_mu, self.d_phase = d_model, d_sigma, d_mu, d_phase
        self.j_max = j_max
        self.kappa = kappa

        self.predict = PredictiveGrounding(d_model, d_sigma, d_phase, **kw)
        self.modulate = GlobalModulate(d_model, d_sigma, d_mu, **kw)
        self.register = Register(d_model, d_sigma, d_mu, **kw)
        self.consistency = Consistency(d_model, d_sigma, d_mu, **kw)
        self.pressure_halt = PressureHalt(d_model, d_sigma, **kw)

        self.condition_out = nn.Linear(d_model, d_model, bias=False, **kw)        # W_o
        self.condition_out_sigma = nn.Linear(d_sigma, d_model, bias=False, **kw)  # W_sigma
        self.condition_norm = nn.LayerNorm(d_model, **kw)

    # ---- §4-§14 helpers (batch-agnostic; operate over any leading (B,L) dims) ----
    def _predict(self, g_t, sigma_prev, phi_prev, evidence_read, no_read):
        q_pred = self.predict.query_proj(sigma_prev)
        r_pred = evidence_read(q_pred, phi_prev, exclude_current=True)
        if no_read:
            r_pred = torch.zeros_like(r_pred)
        pre = (self.predict.read_proj(r_pred) + self.predict.hyp_proj(sigma_prev)
               + self.predict.phase_proj(_phase_embed(phi_prev, self.d_phase)))
        g_hat = self.predict.out_proj(self.predict.norm(pre))
        return g_hat, g_t - g_hat

    def _precision(self, g_t, e, sigma_prev, m_t):
        mu = torch.sigmoid(self.modulate.in_proj_mu(torch.cat([g_t, e, sigma_prev, m_t], -1)))
        e_tilde = mu * self.modulate.err_proj(e)
        return mu, e_tilde

    def _consistency(self, g_t, sigma, r_sigma, mu, sigma_prev):
        dG = self.consistency.P_G(g_t) - self.consistency.Q_G(sigma)
        dR = self.consistency.P_R(r_sigma) - self.consistency.Q_R(sigma)
        muG, muR = self.consistency.prec_G(mu), self.consistency.prec_R(mu)
        return ((muG * dG).pow(2).sum(-1) + (muR * dR).pow(2).sum(-1)
                + self.kappa * (sigma - sigma_prev).pow(2).sum(-1))

    def forward(self, g_t, evidence_read, phi_t, phi_prev=None, sigma_prev=None,
                m_t=None, s_t=None, logits_fn=None, ablation: Optional[str] = None):
        """Revise the hypothesis register and emit the conditioned output o_t (§4-§14).

        g_t: (B,L,d_model). evidence_read(query_512, phi, exclude_current) -> (B,L,d_model).
        phi_t/phi_prev: (B,L,H). sigma_prev: (B,L,d_sigma). m_t: (B,L,d_mu). s_t: (B,L,1).
        Returns (o_t, SilenceAux). Cross-token sigma carry (sigma_prev = prev sigma_bar) is the
        caller's job (sequential decode / truncated training carry) — this is the per-token map.
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
        if ablation == "phase_scrambled":                       # break phase alignment (§22 control)
            phi_t = phi_t.flip(dims=[-2]) if phi_t.shape[-2] > 1 else phi_t

        def read(sigma):
            r = evidence_read(self.register.read_proj(sigma), phi_t, exclude_current=False)
            return torch.zeros_like(r) if no_read else r

        # §4 predictive grounding, §8 precision, §9 register init
        g_hat, e = self._predict(g_t, sigma_prev, phi_prev, evidence_read, no_read)
        mu, e_tilde = self._precision(g_t, e, sigma_prev, m_t)
        sigma0 = self.register.norm(self.register.init_proj(
            torch.cat([sigma_prev, g_t, e_tilde, mu], -1)))

        # §10 revision loop
        sigma_traj, r_traj = [sigma0], []
        sigma_j = sigma0
        for _ in range(self.j_max):
            r_j = read(sigma_j)
            r_traj.append(r_j)
            z = torch.cat([sigma_j, g_t, e_tilde, mu, r_j], -1)
            sigma_j = self.register.norm(
                sigma_j + torch.sigmoid(self.register.update_gate(z))
                * torch.tanh(self.register.update_cand(z)))
            sigma_traj.append(sigma_j)
        r_traj.append(read(sigma_j))                            # read of sigma^Jmax (for E/consistency)

        # §11 consistency over the trajectory
        E_traj = torch.stack([self._consistency(g_t, sigma_traj[j], r_traj[j], mu, sigma_prev)
                              for j in range(self.j_max + 1)], dim=-1)  # (B,L,Jmax+1)

        # §12 integration pressure
        o0 = self._output(g_t, sigma0)
        if logits_fn is not None:
            p0 = torch.softmax(logits_fn(o0), dim=-1)
            H_t = -(p0 * torch.log(p0.clamp_min(1e-9))).sum(-1)
        else:
            H_t = g_t.new_zeros(*lead)
        delta_e = e_tilde.norm(dim=-1)
        delta_sR = (self.consistency.P_R(r_traj[0]) - self.consistency.Q_R(sigma0)).norm(dim=-1)
        zeta = g_t                                              # Pool_zeta = identity (no Appendix params)
        pin = torch.cat([zeta, H_t.unsqueeze(-1), delta_e.unsqueeze(-1),
                         delta_sR.unsqueeze(-1), s_t], -1)
        pi = F.softplus((self.pressure_halt.pressure_out
                         * torch.tanh(self.pressure_halt.pressure_in(pin))).sum(-1))  # (B,L)

        # §13 soft halting (forced p_{Jmax}=1)
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
        sigma_stack = torch.stack(sigma_traj, dim=-2)          # (B,L,Jmax+1,d_sigma)
        sigma_bar = (w.unsqueeze(-1) * sigma_stack).sum(-2)
        expected_J = (w * torch.arange(self.j_max + 1, device=w.device, dtype=w.dtype)).sum(-1)

        if ablation == "no_op":                                 # revision ran but is discarded (§22)
            sigma_bar = sigma0

        # §14 output
        o_t = self._output(g_t, sigma_bar)
        aux = SilenceAux(g_hat, e, mu, e_tilde, sigma0, sigma_traj, r_traj, E_traj, pi, w,
                         expected_J, sigma_bar)
        return o_t, aux

    def _output(self, g_t, sigma):
        return self.condition_out(self.condition_norm(g_t + self.condition_out_sigma(sigma)))
