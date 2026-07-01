# Copyright (c) 2024, Albert Gu and Tri Dao.
"""Pure-PyTorch reference for the AUM-Ø U phase (§6) and the silence read (§10).

Two purposes:
  1. A readable oracle for the resonant AFFINE evidence recurrence
        S_t = alpha_t S_{t-1} + rho_t tau_t (v_hat_t (x) k_rot_t)
     in both a serial (step) and a chunk-parallel form, so the Triton kernel can be
     grad-checked against it and the two forms checked against each other on CPU
     (no Triton/GPU needed).
  2. `aum_state_readout_ref` — the swapped-query readout r = S_t R(phi) q that the
     silence block performs (§4, §10, §12), so silence.py is unit-testable on a Mac.

All functions are float64-capable. The `ssd_minimal_discrete` / `segsum` helpers
(Listing 1 of the Mamba-2 paper) are kept because the chunk reference reuses them.
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# Mamba-2 SSD primitives (reused by the AUM chunk reference)
# ---------------------------------------------------------------------------
def segsum(x):
    """Stable segment-sum: out[..., i, j] = sum_{j < k <= i} x[..., k] (lower-tri)."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None, exclude_diag=False):
    """Linear (affine) SSD: Y_t = C_t sum_{s<=t} exp(sum_{s<r<=t} A_r) B_s X_s.

    X: (b, l, h, p)   A: (b, l, h)   B: (b, l, h, n)   C: (b, l, h, n)
    Returns Y (b, l, h, p) and final_state (b, h, p, n).
    exclude_diag=True gives the STRICTLY-causal read Y_t = C_t sum_{s<t} ... (i.e. reads the
    state S_{t-1}, before the current token's write) — used by the §4 predictive grounding read.
    """
    assert X.shape[1] % block_len == 0
    X, A, B, C = [rearrange(t, "b (c l) ... -> b c l ...", l=block_len) for t in (X, A, B, C)]
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. intra-chunk (diagonal) output
    L = torch.exp(segsum(A))
    if exclude_diag:  # drop the current position (l == s); off-diagonal chunks are already < t
        n = L.shape[-1]
        L = L * (1 - torch.eye(n, device=L.device, dtype=L.dtype))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. per-chunk end states
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. inter-chunk recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. inter-chunk (off-diagonal) output
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


# ---------------------------------------------------------------------------
# AUM-Ø U-phase helpers
# ---------------------------------------------------------------------------
def heavy_tail(x):
    """f(x) = 1 + x for x >= 0 else 1 / (1 - x). The dissolution activation (§6)."""
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


def aum_dynamics(tau_bar, lam_bar, r, theta, dt_bias, eps=1e-4):
    """Map raw controller outputs (§5) to (tau, alpha_log, rho, dphi) (§6).

    tau_bar, lam_bar, r, theta, dt_bias broadcast to (..., H).
    Returns tau (>0), alpha_log = -lambda*tau (<=0), rho in (0,1), dphi = pi*tanh(theta)*tau.
    """
    tau = F.softplus(tau_bar + dt_bias)
    lam = eps + heavy_tail(lam_bar)
    alpha_log = -lam * tau
    rho = torch.sigmoid(r)
    dphi = math.pi * torch.tanh(theta) * tau
    return tau, alpha_log, rho, dphi


def ladder_freqs(n_blocks, omega_min=1e-3, omega_max=1.0, device=None, dtype=None):
    """The geometric frequency ladder omega_b = omega_max (omega_min/omega_max)^{(b-1)/(B-1)} (§4).

    Returns (n_blocks,), descending from omega_max to omega_min. Fixed (a buffer, not a parameter).
    """
    dtype = dtype or torch.float32
    if n_blocks == 1:
        return torch.full((1,), omega_max, device=device, dtype=dtype)
    b = torch.arange(n_blocks, device=device, dtype=dtype)
    return omega_max * (omega_min / omega_max) ** (b / (n_blocks - 1))


def _rotate_ladder(x, phi, freqs=None):
    """Multi-frequency rotation R(phi) (§4): rotate the b-th adjacent pair of the last dim by
    omega_b * phi. x: (..., D) with D even; phi: (...,) broadcast; freqs: (D/2,) — the geometric
    ladder by default. Orthogonal per token, so read/write scores depend only on relative phase;
    the ladder makes alignment decay quasi-monotonically in |Δphi| instead of ringing at 2π.
    """
    B = x.shape[-1] // 2
    if freqs is None:
        freqs = ladder_freqs(B, device=x.device)
    xr = x.reshape(*x.shape[:-1], B, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    ang = phi.unsqueeze(-1) * freqs                              # (..., D/2)
    cos, sin = torch.cos(ang), torch.sin(ang)
    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos
    return torch.stack([r0, r1], dim=-1).reshape_as(x)


def _l2norm(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _gated_rmsnorm(x, z, weight=None, eps=1e-5):
    """silu(z) ⊙ RMSNorm(x) over the last dim (the U-phase readout gate, §6)."""
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        x = x * weight
    return x * (z * torch.sigmoid(z))


def aum_unfold_step_ref(q, k, v, tau_bar, lam_bar, r, theta, z=None, D=None,
                        dt_bias=0.0, eps=1e-4, S0=None, phi0=None, norm_weight=None, freqs=None):
    """Serial (recurrent) reference for the §4 U phase.

    q, k: (B, L, H, Dqk)   v, z: (B, L, H, Dv)
    tau_bar, lam_bar, r, theta: (B, L, H)   D: (H,) or (H, Dv)   freqs: (Dqk/2,) ladder (§4).
    Returns h_U (B, L, H, Dv), and final (S_T (B,H,Dv,Dqk), phi_T (B,H)).
    """
    B, L, H, Dqk = q.shape
    Dv = v.shape[-1]
    dtype = q.dtype
    S = torch.zeros(B, H, Dv, Dqk, dtype=dtype, device=q.device) if S0 is None else S0.clone()
    phi = torch.zeros(B, H, dtype=dtype, device=q.device) if phi0 is None else phi0.clone()

    outs = []
    for t in range(L):
        tau, alpha_log, rho, dphi = aum_dynamics(
            tau_bar[:, t], lam_bar[:, t], r[:, t], theta[:, t], dt_bias, eps
        )
        phi = phi + dphi
        k_hat = _l2norm(k[:, t])
        v_hat = _l2norm(v[:, t])
        q_rot = _rotate_ladder(q[:, t], phi, freqs)
        k_rot = _rotate_ladder(k_hat, phi, freqs)
        w = (rho * tau).unsqueeze(-1).unsqueeze(-1)             # (B,H,1,1)
        S = torch.exp(alpha_log).unsqueeze(-1).unsqueeze(-1) * S
        S = S + w * (v_hat.unsqueeze(-1) * k_rot.unsqueeze(-2))  # v_hat (x) k_rot -> (B,H,Dv,Dqk)
        out = torch.einsum("bhpn,bhn->bhp", S, q_rot)            # S_t q_rot
        if D is not None:
            Dv_ = D if D.dim() == 2 else D.unsqueeze(-1)         # (H,Dv) or (H,1)
            out = out + Dv_ * v[:, t]
        outs.append(out)
    h = torch.stack(outs, dim=1)                                 # (B,L,H,Dv)
    if z is not None:
        h = _gated_rmsnorm(h, z, norm_weight)
    return h, (S, phi)


def aum_unfold_chunk_ref(q, k, v, tau_bar, lam_bar, r, theta, z=None, D=None,
                         dt_bias=0.0, eps=1e-4, block_len=None, norm_weight=None, freqs=None):
    """Chunk-parallel reference for the §4 U phase (segsum factorization).

    Same args/returns as aum_unfold_step_ref (without initial states). This is the
    oracle the Triton kernel reproduces; it must match aum_unfold_step_ref exactly.
    """
    B, L, H, Dqk = q.shape
    Dv = v.shape[-1]
    if block_len is None:
        block_len = L
    tau, alpha_log, rho, dphi = aum_dynamics(tau_bar, lam_bar, r, theta, dt_bias, eps)
    phi = torch.cumsum(dphi, dim=1)                              # (B,L,H)
    q_rot = _rotate_ladder(q, phi, freqs)
    k_rot = _rotate_ladder(_l2norm(k), phi, freqs)
    X = (rho * tau).unsqueeze(-1) * _l2norm(v)                   # w * v_hat  (B,L,H,Dv)
    Y, final_state = ssd_minimal_discrete(X, alpha_log, k_rot, q_rot, block_len)
    if D is not None:
        Dv_ = D if D.dim() == 2 else D.unsqueeze(-1)
        Y = Y + Dv_ * v
    if z is not None:
        Y = _gated_rmsnorm(Y, z, norm_weight)
    return Y, (final_state, phi[:, -1])


def aum_state_readout_ref(query, k, v, tau_bar, lam_bar, r, theta, phi=None,
                          dt_bias=0.0, eps=1e-4, block_len=None, exclude_current=False,
                          freqs=None, rotate_query=True):
    """Swapped-query readout r_t = S_t R(phi_t) query_t (the silence read, §5/§8).

    Reuses the SAME evidence write (k, v, dynamics) as the U phase, but reads with an
    external `query` (B, L, H, Dqk) instead of the token's own q. Returns r (B, L, H, Dv).
    If `phi` (B,L,H) is supplied it is reused; otherwise it is recomputed from the dynamics.
    exclude_current=True reads S_{t-1} (the §5 predictive read). rotate_query=False applies the
    raw query (a phase-free read: with query = 1/Dqk this is Pool(S), the §14 pooled-evidence
    read of the Top-GRU baseline).
    """
    B, L, H, Dqk = query.shape
    if block_len is None:
        block_len = L
    tau, alpha_log, rho, dphi = aum_dynamics(tau_bar, lam_bar, r, theta, dt_bias, eps)
    if phi is None:
        phi = torch.cumsum(dphi, dim=1)
    q_rot = _rotate_ladder(query, phi, freqs) if rotate_query else query
    k_rot = _rotate_ladder(_l2norm(k), phi, freqs)
    X = (rho * tau).unsqueeze(-1) * _l2norm(v)
    r_out, _ = ssd_minimal_discrete(X, alpha_log, k_rot, q_rot, block_len, exclude_diag=exclude_current)
    if exclude_current:
        # The decay tile decays to position i; reading S_{t-1} should decay only to i-1, so undo
        # the extra alpha_i = exp(alpha_log_i) step.
        r_out = r_out * torch.exp(-alpha_log).unsqueeze(-1)
    return r_out
