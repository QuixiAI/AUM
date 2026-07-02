# Metal path for the fused sequential global block (v6 §5-§9; kernel-roadmap step 4).
#
# Pack builders mirror the layouts hardcoded in kernels/metal/src/aum_silence.metal:
#   * build_weight_pack : one flat fp32 tensor, torch.cat of the module's weight column blocks in
#     the kernel's W_* offset order. Built INSIDE the autograd graph, so the backward kernel's
#     d(wpack) routes to the parameters through this cat.
#   * build_stream_pack : (B, L, SW) — the token-parallel precompute streams + g + cos/sin.
#
# unpack_save() reslices the kernel's per-token save pack into the named trajectories.

import torch
import torch.nn.functional as F

from aum_ssm.modules.silence import _phase_embed
from aum_ssm.ops.metal.silence_flat import _cols

SW, SV, J = 2720, 3151, 2                       # must match aum_silence.metal


def build_weight_pack(silence):
    """Flat fp32 weight pack in the kernel's offset order (differentiable cat)."""
    d, s, m = silence.d_model, silence.d_sigma, silence.d_mu
    _, mu_e, mu_s, _ = _cols(silence.modulate.in_proj_mu.weight, d, d, s, m)
    i_s, _, i_et, i_mu = _cols(silence.register.init_proj.weight, s, d, m, m)
    g_s, _, g_et, g_mu, g_r = _cols(silence.register.update_gate.weight, s, d, m, m, d)
    n_s, _, n_et, n_mu, n_r = _cols(silence.register.update_cand.weight, s, d, m, m, d)
    wp = silence.pressure_halt.pressure_in.weight
    h1_s, h1_pi, h1_e = _cols(silence.pressure_halt.halt_1.weight, s, 1, 1)
    parts = [
        silence.predict.query_proj.weight,      # W_QPRED (D, DS)
        silence.predict.read_proj.weight,       # W_R     (D, D)
        silence.predict.hyp_proj.weight,        # W_HYP   (D, DS)
        silence.predict.out_proj.weight,        # W_P     (D, D)
        silence.predict.norm.weight,            # LN5 w
        silence.predict.norm.bias,              # LN5 b
        mu_e, mu_s,                             # W_MUE, W_MUS
        silence.modulate.err_proj.weight,       # W_ERR
        i_s, i_et, i_mu,                        # W_IS, W_IET, W_IMU
        silence.register.norm.weight,           # LN1 w
        silence.register.norm.bias,             # LN1 b
        silence.register.read_proj.weight,      # W_QSIG (D, DS)
        g_s, g_et, g_mu, g_r,                   # W_G*
        n_s, n_et, n_mu, n_r,                   # W_N*
        silence.consistency.Q_G.weight,         # W_QG
        silence.consistency.P_R.weight,         # W_PRR
        silence.consistency.Q_R.weight,         # W_QR
        silence.consistency.prec_G.weight,      # W_PCG
        silence.consistency.prec_R.weight,      # W_PCR
        silence.pressure_halt.pressure_out,     # W_WPI
        wp[:, d], wp[:, d + 1],                 # W_WPDE, W_WPDS (scalar feature columns)
        h1_s, h1_pi[:, 0], h1_e[:, 0],          # W_H1S, W_H1PI, W_H1E
        silence.pressure_halt.halt_2.weight[0],  # W_H2
    ]
    return torch.cat([p.reshape(-1) for p in parts]).float()


def build_stream_pack(silence, g, phi, m_t, s_t, freqs):
    """(B, L, SW): the token-parallel streams, in the kernel's SW_* order (differentiable)."""
    B, L, H = phi.shape
    d, s, m = silence.d_model, silence.d_sigma, silence.d_mu
    Wmu_g, _, _, Wmu_m = _cols(silence.modulate.in_proj_mu.weight, d, d, s, m)
    Wi_g = _cols(silence.register.init_proj.weight, s, d, m, m)[1]
    Wg_g = _cols(silence.register.update_gate.weight, s, d, m, m, d)[1]
    Wn_g = _cols(silence.register.update_cand.weight, s, d, m, m, d)[1]
    Wp = silence.pressure_halt.pressure_in.weight
    phi_prev = torch.cat([phi.new_zeros(B, 1, H), phi[:, :-1]], dim=1)
    ang_t = (phi.unsqueeze(-1) * freqs).reshape(B, L, -1)
    ang_p = (phi_prev.unsqueeze(-1) * freqs).reshape(B, L, -1)
    return torch.cat([
        F.linear(g, Wmu_g) + F.linear(m_t, Wmu_m),                    # c_mu     (32)
        F.linear(g, Wi_g),                                            # c_init   (128)
        F.linear(g, Wg_g),                                            # c_gate   (128)
        F.linear(g, Wn_g),                                            # c_cand   (128)
        silence.consistency.P_G(g),                                   # c_pg     (128)
        F.linear(g, Wp[:, :d]) + Wp[:, d + 2] * s_t,                  # c_press  (128)
        silence.predict.phase_proj(_phase_embed(phi_prev, silence.d_phase)),  # c_phase (512)
        g,                                                            # g        (512)
        torch.cos(ang_t), torch.sin(ang_t),                           # cos/sin_t (256 each)
        torch.cos(ang_p), torch.sin(ang_p),                           # cos/sin_p (256 each)
    ], dim=-1).float()


def unpack_save(save):
    """Reslice the (B, L, SV) save pack into named fp32 trajectories."""
    D, DS, DM = 512, 128, 32
    off, out = 0, {}
    for name, width in [("g_hat", D), ("mu", DM), ("e_tilde", DM),
                        ("sigma_stack", (J + 1) * DS), ("r_pred", D), ("r_stack", (J + 1) * D),
                        ("E", J + 1), ("pi", 1), ("w", J + 1), ("sigma_star", DS),
                        ("ln5", 2), ("ln1", 2 * (J + 1))]:
        out[name] = save[..., off:off + width]
        off += width
    assert off == SV, off
    B, L = save.shape[:2]
    out["sigma_stack"] = out["sigma_stack"].reshape(B, L, J + 1, DS)
    out["r_stack"] = out["r_stack"].reshape(B, L, J + 1, D)
    out["pi"] = out["pi"].squeeze(-1)
    return out


def silence_fwd_metal(silence, g, phi, m_t, s_t, alpha, xw, k_rot, freqs,
                      halt_u=None, forced_depth=None):
    """Run the fused forward. Returns (unpacked trajectories dict, j_star, save, packs)."""
    import kernels.metal as km
    streams = build_stream_pack(silence, g, phi, m_t, s_t, freqs)
    wpack = build_weight_pack(silence)
    B, L = g.shape[:2]
    hu = halt_u if halt_u is not None else g.new_zeros(B, L)
    forced = -1 if forced_depth is None else int(forced_depth)
    save, j_star, S_final, S_ckpt = km.aum_silence_fwd(
        streams.detach(), alpha.detach().float(),
        xw.detach().float().reshape(B, L, -1), k_rot.detach().float().reshape(B, L, -1),
        hu.float(), wpack.detach(), silence.kappa, forced)
    out = unpack_save(save)
    return out, j_star, save, (streams, wpack, S_ckpt)
