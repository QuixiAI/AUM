# AUM-Ø M phase: error-free precision modulation, per evidence layer (§8).
#
#   mu^l   = sigma(W_mu^l [h^A, h^U, m])
#   dh^l   = U_m diag(mu^l) V_m h^U
#   h^M^l  = h^A + h^U + dh^l
#
# Low-rank, error-free below the top (no e_t). Maps to model.layers.*.modulate.* in Appendix A.

import torch
import torch.nn as nn


class PrecisionModulate(nn.Module):
    def __init__(self, d_model, d_mu=32, m_dim=32, device=None, dtype=None):
        kw = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_mu = d_mu
        # in_proj_mu input = [h^A (d_model), h^U (d_model), m (m_dim)]
        self.in_proj_mu = nn.Linear(2 * d_model + m_dim, d_mu, bias=False, **kw)  # W_mu
        self.down = nn.Linear(d_model, d_mu, bias=False, **kw)                    # V_m
        self.up = nn.Linear(d_mu, d_model, bias=False, **kw)                      # U_m

    def forward(self, h_A, h_U, m):
        """Returns h^M = h^A + h^U + dh (§8)."""
        mu = torch.sigmoid(self.in_proj_mu(torch.cat([h_A, h_U, m], dim=-1)))  # (..., d_mu) in [0,1]
        dh = self.up(mu * self.down(h_U))                                       # U_m diag(mu) V_m h^U
        return h_A + h_U + dh, mu
