# AUM-Ø global silence block (§3-§14). The entire net-new subsystem.
#
# This is a PARAMETER SKELETON: the submodules and their shapes mirror Appendix A of AUM-Ø.md
# exactly (model.silence.*), so the state_dict layout is correct, but forward() is not yet
# implemented. The silence loop is plain PyTorch (frozen S_t, tiny J_max unroll) — no custom kernel.
#
# TODO(AUM): implement, in order:
#   1. predictive grounding  -> g_hat_t, e_t              (§4)
#   2. error-fed precision    -> mu_t, e_tilde_t          (§8)
#   3. register init          -> sigma_t^0                (§9)
#   4. revision loop (J_max)  -> sigma_t^{j+1}            (§10)
#   5. consistency functional -> E_t(sigma)               (§11)
#   6. integration pressure   -> pi_t                     (§12)
#   7. soft halting           -> w_j, sigma_bar           (§13)
#   8. output / condition     -> o_t                      (§14)

import torch
import torch.nn as nn


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
        # W_mu input = [g (d_model), e (d_model), sigma (d_sigma), m (d_mu)]
        self.in_proj_mu = nn.Linear(2 * d_model + d_sigma + d_mu, d_mu, bias=False, **kw)


class Register(nn.Module):
    """Bottlenecked hypothesis register: init + nonlinear gated revision loop (§9, §10)."""

    def __init__(self, d_model, d_sigma, d_mu, device=None, dtype=None):
        kw = {"device": device, "dtype": dtype}
        super().__init__()
        # init input = [sigma (d_sigma), g (d_model), e_tilde (d_mu), mu (d_mu)]
        self.init_proj = nn.Linear(d_sigma + d_model + 2 * d_mu, d_sigma, bias=False, **kw)  # W_sigma0
        self.read_proj = nn.Linear(d_sigma, d_model, bias=False, **kw)    # W_q^sigma: sigma -> state-key space
        # update input = [sigma, g, e_tilde, mu, r^j (d_model)]
        upd_in = d_sigma + d_model + 2 * d_mu + d_model
        self.update_gate = nn.Linear(upd_in, d_sigma, bias=False, **kw)   # W_g
        self.update_cand = nn.Linear(upd_in, d_sigma, bias=False, **kw)   # W_n
        self.norm = nn.LayerNorm(d_sigma, **kw)


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
        # pressure_in input = [zeta (d_model), H_t, Delta^e, Delta^{sigma R}, s_t] = d_model + 4
        self.pressure_in = nn.Linear(d_model + 4, d_sigma, bias=False, **kw)
        self.pressure_out = nn.Parameter(torch.zeros(d_sigma, **kw))      # w_pi (vector)
        # halt input = [sigma (d_sigma), pi_t, E_t] = d_sigma + 2
        self.halt_1 = nn.Linear(d_sigma + 2, halt_hidden, bias=False, **kw)
        self.halt_2 = nn.Linear(halt_hidden, 1, bias=False, **kw)


class SilenceBlock(nn.Module):
    """The single global silence block on top of the evidence stack (§3).

    Reads the top evidence layer's S_t and phi_t, revises a global hypothesis register sigma when
    expected benefit is high, and folds it into the output. ~1.8M params for the Tiny reference.
    """

    def __init__(self, d_model, d_sigma=128, d_mu=32, d_phase=32, j_max=2, device=None, dtype=None):
        kw = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_sigma = d_sigma
        self.d_mu = d_mu
        self.d_phase = d_phase
        self.j_max = j_max

        self.predict = PredictiveGrounding(d_model, d_sigma, d_phase, **kw)
        self.modulate = GlobalModulate(d_model, d_sigma, d_mu, **kw)
        self.register = Register(d_model, d_sigma, d_mu, **kw)
        self.consistency = Consistency(d_model, d_sigma, d_mu, **kw)
        self.pressure_halt = PressureHalt(d_model, d_sigma, **kw)

        # Output / condition projection (§14): o_t = W_o LN(g_t + W_sigma sigma_bar)
        self.condition_out = nn.Linear(d_model, d_model, bias=False, **kw)        # W_o
        self.condition_out_sigma = nn.Linear(d_sigma, d_model, bias=False, **kw)  # W_sigma
        self.condition_norm = nn.LayerNorm(d_model, **kw)

    def forward(self, g_t, S_t=None, phi_t=None, sigma_prev=None, inference_params=None):
        """Revise the hypothesis register and emit the conditioned output o_t (§4-§14).

        TODO(AUM): not yet implemented. Requires the top evidence layer to expose S_t and phi_t,
        and returns (o_t, aux) where aux carries e_t, mu_t, pi_t, J_t, sigma_bar for the losses (§18)
        and diagnostics (§21).
        """
        raise NotImplementedError(
            "SilenceBlock.forward is a skeleton; implement the §4-§14 silence loop. "
            "Backbone runs with silence_enabled=False until then."
        )
