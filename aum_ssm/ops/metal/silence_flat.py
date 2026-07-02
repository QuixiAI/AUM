# The "flat" silence-block recurrence — the kernel ABI, in PyTorch (v6 §5-§9, roadmap step 4).
#
# AumBackbone's global recurrence runs SilenceBlock once per token: a Python loop of ~50 tiny MPS
# ops per step, launch-bound (~90% of a train step). This module reorganizes the SAME math into
# the shape the fused Metal kernel computes:
#
#   * PRECOMPUTED STREAMS (token-parallel, batched GEMMs outside the loop): every column block of
#     every concat-Linear that multiplies a quantity known for all tokens up front — the g_t / m_t
#     parts of mu/init/gate/cand/pressure, P_G g_t, the phase-embed projection W_phi Phi(phi_{t-1}),
#     and the rotation-ladder cos/sin at phi_t and phi_{t-1}.
#   * A THIN SEQUENTIAL CORE per token: the evidence-state step S_t = alpha_t*S_{t-1} + x_t (x) k_t,
#     four S-reads, and the small matvecs against the sigma/e/mu/r column blocks.
#
# `flat_forward` runs that core as a PyTorch loop — it must match `_global_segment` EXACTLY (same
# ops, same order, fp32); it is the oracle the Metal kernels (fwd + bwd) are validated against,
# and its stream/weight-slice layout IS the kernel's buffer layout.
#
# Scope (the standard training path; everything else falls back to the module loop):
#   top_gru=False, ablation=None, entropy_feature=False, pi_trigger unused (training),
#   halt_u given (training) or forced_depth given (stage 2 / labels).

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from aum_ssm.modules.silence import _phase_embed


@dataclass
class SilenceStreams:
    """Token-parallel precomputes. Shapes: (B, L, ...); built by big batched GEMMs."""
    c_mu: torch.Tensor        # W_mu[:, g] g + W_mu[:, m] m                     (B,L,32)
    c_init: torch.Tensor      # W_sigma0[:, g] g                                (B,L,128)
    c_gate: torch.Tensor      # W_g[:, g] g                                     (B,L,128)
    c_cand: torch.Tensor      # W_n[:, g] g                                     (B,L,128)
    c_pg: torch.Tensor        # P_G g                                           (B,L,128)
    c_press: torch.Tensor     # pressure_in[:, g] g + pressure_in[:, s] s       (B,L,128)
    c_phase: torch.Tensor     # W_phi Phi(phi_{t-1})                            (B,L,512)
    cos_t: torch.Tensor       # cos(freqs * phi_t)  per head                    (B,L,H,Dh/2)
    sin_t: torch.Tensor
    cos_p: torch.Tensor       # cos(freqs * phi_{t-1}) per head (row 0: phi=0)  (B,L,H,Dh/2)
    sin_p: torch.Tensor


def _cols(w, *sizes):
    """Split a concat-Linear weight (out, sum(sizes)) into per-input column blocks."""
    out, off = [], 0
    for s in sizes:
        out.append(w[:, off:off + s])
        off += s
    assert off == w.shape[1], (off, w.shape)
    return out


def precompute_streams(silence, g, phi, m_t, s_t, freqs):
    """All token-parallel inputs of the sequential core (batched GEMMs; no loop)."""
    B, L, H = phi.shape
    d_model, d_sigma, d_mu = silence.d_model, silence.d_sigma, silence.d_mu
    Wmu_g, _, _, Wmu_m = _cols(silence.modulate.in_proj_mu.weight, d_model, d_model, d_sigma, d_mu)
    Wi_g = _cols(silence.register.init_proj.weight, d_sigma, d_model, d_mu, d_mu)[1]
    Wg_g = _cols(silence.register.update_gate.weight, d_sigma, d_model, d_mu, d_mu, d_model)[1]
    Wn_g = _cols(silence.register.update_cand.weight, d_sigma, d_model, d_mu, d_mu, d_model)[1]
    Wp = silence.pressure_halt.pressure_in.weight            # (128, 512+3)
    Wp_g, Wp_scal = Wp[:, :d_model], Wp[:, d_model:]         # scalar cols: [de, dsR, s]

    phi_prev = torch.cat([phi.new_zeros(B, 1, H), phi[:, :-1]], dim=1)
    ang_t = phi.unsqueeze(-1) * freqs                        # (B,L,H,Dh/2)
    ang_p = phi_prev.unsqueeze(-1) * freqs
    return SilenceStreams(
        c_mu=F.linear(g, Wmu_g) + F.linear(m_t, Wmu_m),
        c_init=F.linear(g, Wi_g),
        c_gate=F.linear(g, Wg_g),
        c_cand=F.linear(g, Wn_g),
        c_pg=silence.consistency.P_G(g),
        c_press=F.linear(g, Wp_g) + Wp_scal[:, 2] * s_t,     # s_t is (B,L,1)
        c_phase=silence.predict.phase_proj(_phase_embed(phi_prev, silence.d_phase)),
        cos_t=torch.cos(ang_t), sin_t=torch.sin(ang_t),
        cos_p=torch.cos(ang_p), sin_p=torch.sin(ang_p),
    )


def _rot_read(S, q, cos, sin, H, Dh):
    """q (B,d_model) -> per-head ladder rotation at (cos,sin) (B,H,Dh/2) -> r = S q (B,d_model)."""
    B = q.shape[0]
    qh = q.reshape(B, H, Dh // 2, 2)
    q0, q1 = qh[..., 0], qh[..., 1]
    qr = torch.stack([q0 * cos - q1 * sin, q0 * sin + q1 * cos], dim=-1).reshape(B, H, Dh)
    return torch.einsum("bhpn,bhn->bhp", S, qr).reshape(B, H * Dh)


def flat_forward(silence, streams, g, m_t, alpha, xw, k_rot, halt_u=None, forced_depth=None,
                 S0=None, sigma_carry=None):
    """The thin sequential core over all L tokens (PyTorch loop — the kernel oracle).

    g (B,L,512); m_t (B,L,32); alpha (B,L,H); xw (B,L,H,Dv); k_rot (B,L,H,Dk);
    halt_u (B,L) or None (then forced_depth must be given). Returns a dict of the
    sequence-shaped trajectories the kernel emits (fp32), plus the final carries.
    """
    B, L, H, Dv = xw.shape
    d_model, d_sigma, d_mu, J = silence.d_model, silence.d_sigma, silence.d_mu, silence.j_max
    Dh = Dv
    dev = g.device
    S = S0 if S0 is not None else g.new_zeros(B, H, Dv, k_rot.shape[-1])
    sigma = sigma_carry if sigma_carry is not None else g.new_zeros(B, d_sigma)

    # weight column blocks of the sequential core
    _, Wmu_e, Wmu_s, _ = _cols(silence.modulate.in_proj_mu.weight, d_model, d_model, d_sigma, d_mu)
    Wi = _cols(silence.register.init_proj.weight, d_sigma, d_model, d_mu, d_mu)
    Wi_s, Wi_et, Wi_mu = Wi[0], Wi[2], Wi[3]
    Wg = _cols(silence.register.update_gate.weight, d_sigma, d_model, d_mu, d_mu, d_model)
    Wn = _cols(silence.register.update_cand.weight, d_sigma, d_model, d_mu, d_mu, d_model)
    Wp_scal = silence.pressure_halt.pressure_in.weight[:, d_model:]
    Wh1 = _cols(silence.pressure_halt.halt_1.weight, d_sigma, 1, 1)
    ln5, ln1 = silence.predict.norm, silence.register.norm

    outs = {k: [] for k in ("g_hat", "mu", "e_tilde", "sigma_stack", "r_pred", "r_stack",
                            "E", "pi", "w", "j_star", "sigma_star")}
    for t in range(L):
        S_prev = S
        S = (alpha[:, t].unsqueeze(-1).unsqueeze(-1) * S_prev
             + xw[:, t].unsqueeze(-1) * k_rot[:, t].unsqueeze(-2))

        # §5 predictive grounding (reads S_{t-1} at phi_{t-1})
        q_pred = silence.predict.query_proj(sigma)
        r_pred = _rot_read(S_prev, q_pred, streams.cos_p[:, t], streams.sin_p[:, t], H, Dh)
        pre = (silence.predict.read_proj(r_pred) + silence.predict.hyp_proj(sigma)
               + streams.c_phase[:, t])
        g_hat = silence.predict.out_proj(ln5(pre))
        e = g[:, t] - g_hat

        # §6 precision
        mu = torch.sigmoid(streams.c_mu[:, t] + F.linear(e, Wmu_e) + F.linear(sigma, Wmu_s))
        e_tilde = mu * silence.modulate.err_proj(e)

        # §7 register init
        sigma0 = ln1(streams.c_init[:, t] + F.linear(sigma, Wi_s) + F.linear(e_tilde, Wi_et)
                     + F.linear(mu, Wi_mu))

        # §8 revision loop (reads S_t at phi_t)
        sig_traj, r_traj = [sigma0], []
        sj = sigma0
        for _ in range(J):
            rj = _rot_read(S, silence.register.read_proj(sj), streams.cos_t[:, t],
                           streams.sin_t[:, t], H, Dh)
            gate = torch.sigmoid(streams.c_gate[:, t] + F.linear(sj, Wg[0])
                                 + F.linear(e_tilde, Wg[2]) + F.linear(mu, Wg[3])
                                 + F.linear(rj, Wg[4]))
            cand = torch.tanh(streams.c_cand[:, t] + F.linear(sj, Wn[0])
                              + F.linear(e_tilde, Wn[2]) + F.linear(mu, Wn[3])
                              + F.linear(rj, Wn[4]))
            sj = ln1(sj + gate * cand)
            r_traj.append(rj)
            sig_traj.append(sj)
        r_traj.append(_rot_read(S, silence.register.read_proj(sj), streams.cos_t[:, t],
                                streams.sin_t[:, t], H, Dh))

        # §7 consistency (mu detached), §9 pressure, §8 halting
        mu_d = mu.detach()
        muG, muR = silence.consistency.prec_G(mu_d), silence.consistency.prec_R(mu_d)
        E = []
        for j in range(J + 1):
            dG = streams.c_pg[:, t] - silence.consistency.Q_G(sig_traj[j])
            dR = silence.consistency.P_R(r_traj[j]) - silence.consistency.Q_R(sig_traj[j])
            E.append((muG * dG).pow(2).sum(-1) + (muR * dR).pow(2).sum(-1)
                     + silence.kappa * (sig_traj[j] - sigma).pow(2).sum(-1))
        E = torch.stack(E, dim=-1)                            # (B, J+1)
        de = e_tilde.norm(dim=-1)
        dsR = (silence.consistency.P_R(r_traj[0]) - silence.consistency.Q_R(sigma0)).norm(dim=-1)
        pre_pi = streams.c_press[:, t] + Wp_scal[:, 0] * de.unsqueeze(-1) \
            + Wp_scal[:, 1] * dsR.unsqueeze(-1)
        pi = F.softplus((silence.pressure_halt.pressure_out * torch.tanh(pre_pi)).sum(-1))

        ps = []
        for j in range(J + 1):
            if j < J:
                h = torch.tanh(F.linear(sig_traj[j], Wh1[0]) + Wh1[1][:, 0] * pi.unsqueeze(-1)
                               + Wh1[2][:, 0] * E[:, j:j + 1])
                ps.append(torch.sigmoid(silence.pressure_halt.halt_2(h)).squeeze(-1))
            else:
                ps.append(torch.ones_like(pi))
        w, cum = [], torch.ones_like(pi)
        for j in range(J + 1):
            w.append(ps[j] * cum)
            cum = cum * (1 - ps[j])
        w = torch.stack(w, dim=-1)

        if forced_depth is not None:
            j_star = torch.full((B,), int(forced_depth), device=dev, dtype=torch.long)
            w = F.one_hot(j_star, J + 1).to(w.dtype)
        else:
            hit = (w.cumsum(-1) >= halt_u[:, t].unsqueeze(-1)).to(torch.uint8)
            j_star = hit.argmax(dim=-1)

        sig_stack = torch.stack(sig_traj, dim=1)              # (B, J+1, d_sigma)
        sigma = sig_stack.gather(1, j_star[:, None, None].expand(B, 1, d_sigma)).squeeze(1)

        outs["g_hat"].append(g_hat); outs["mu"].append(mu); outs["e_tilde"].append(e_tilde)
        outs["sigma_stack"].append(sig_stack); outs["r_pred"].append(r_pred)
        outs["r_stack"].append(torch.stack(r_traj, dim=1)); outs["E"].append(E)
        outs["pi"].append(pi); outs["w"].append(w); outs["j_star"].append(j_star)
        outs["sigma_star"].append(sigma)

    res = {k: torch.stack(v, dim=1) for k, v in outs.items()}
    res["S_final"], res["sigma_final"] = S, sigma
    return res
