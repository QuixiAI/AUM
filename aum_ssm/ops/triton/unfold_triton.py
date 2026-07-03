# Triton/CUDA path for the AUM-Ø U phase (§4-§6) — nvidia-plan B2/B3.
#
# The U-phase IS mamba2 SSD with pre-folded operands (C = R(phi)q, B = R(phi)k_hat,
# X = rho*tau*v_hat, per-token log-decay alpha_log = -lambda*tau), so the core routes to Tri
# Dao's tuned chunked-SSD Triton kernels (vendored in aum_ssm/ops/triton/ssd_*.py) — measured
# faster on sm_86 than porting the Metal chunked pipeline (see AUM-nvidia-plan.md B3: route by
# measurement).
#
# The cumlog <-> dt*A parameterization adaptation: upstream couples the decay (dA = dt*A) and
# the input scale (x_t * dt_t) through one dt. AUM uses DIFFERENT per-token scalars — decay
# exp(-lambda*tau) vs write scale rho*tau — so _AumSSDCore drives the INTERNAL kernels
# directly: dt := rho*tau enters only as the input scale, dA_cumsum := within-chunk cumsum of
# alpha_log enters only as the decay. The backward mirrors _mamba_chunk_scan_combined_bwd but
# stops before _chunk_cumsum_bwd: the assembled per-token ddA IS d(alpha_log) and ddt IS
# d(rho*tau), no (dt, A) recoupling.
#
# The epilogue (silu(z) * RMSNorm_perhead(Y + D_skip*v) * norm_weight) reuses the vendored
# layernorm_gated Triton kernels (group_size=headdim, norm_before_gate=True).

import torch
from einops import rearrange

from aum_ssm.modules.ssd_reference import _l2norm, _rotate_ladder, aum_dynamics
from aum_ssm.ops.triton.layernorm_gated import rmsnorm_fn
from aum_ssm.ops.triton.ssd_bmm import _bmm_chunk_bwd, _bmm_chunk_fwd
from aum_ssm.ops.triton.ssd_chunk_scan import (_chunk_scan_bwd_dC, _chunk_scan_bwd_dcb,
                                               _chunk_scan_bwd_ddAcs_stable,
                                               _chunk_scan_bwd_dstates, _chunk_scan_fwd)
from aum_ssm.ops.triton.ssd_chunk_state import _chunk_state_bwd_db, _chunk_state_fwd
from aum_ssm.ops.triton.ssd_combined import _chunk_scan_chunk_state_bwd_dx
from aum_ssm.ops.triton.ssd_state_passing import _state_passing_bwd, _state_passing_fwd


class _AumSSDCore(torch.autograd.Function):
    """Y_t = C_t . sum_{s<=t} exp(cumlog_t - cumlog_s) (B_s . X_s), X = dt (.) x — the AUM
    evidence recurrence over the upstream chunked-SSD internals (chunk 64, linear time).

    x/B/C (b, l, h, d) in the compute dtype (bf16 under autocast); dt, alpha_log (b, l, h)
    fp32. l % chunk_size == 0 (the caller falls back to the reference otherwise)."""

    @staticmethod
    def forward(ctx, x, dt, alpha_log, B, C, chunk_size):
        b, l, h, dstate = B.shape
        assert l % chunk_size == 0
        x, B, C = x.contiguous(), B.contiguous(), C.contiguous()
        dt_hl = rearrange(dt, "b (c s) h -> b h c s", s=chunk_size).float().contiguous()
        dA_cs = rearrange(alpha_log, "b (c s) h -> b h c s",
                          s=chunk_size).float().cumsum(dim=-1).contiguous()
        states = _chunk_state_fwd(B, x, dt_hl, dA_cs, states_in_fp32=True)
        states, _final = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"),
                                            dA_cs[:, :, :, -1], chunk_size=chunk_size,
                                            out_dtype=C.dtype)
        states = rearrange(states, "... (p n) -> ... p n", n=dstate)
        CB = _bmm_chunk_fwd(C, B, chunk_size, output_dtype=torch.float32)
        out, _out_x = _chunk_scan_fwd(CB, x, dt_hl, dA_cs, C, states, D=None, z=None)
        ctx.save_for_backward(x, dt_hl, dA_cs, B, C)
        ctx.chunk_size = chunk_size
        return out

    @staticmethod
    def backward(ctx, dout):
        x, dt_hl, dA_cs, B, C = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        b, l, h, dstate = B.shape
        dout = dout.contiguous()
        # recompute CB and the scanned states (memory over FLOPs, like the upstream backward)
        CB = _bmm_chunk_fwd(C, B, chunk_size, output_dtype=torch.float32)
        states = _chunk_state_fwd(B, x, dt_hl, dA_cs, states_in_fp32=True)
        states, _ = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"),
                                       dA_cs[:, :, :, -1], chunk_size=chunk_size)
        states = rearrange(states, "... (p n) -> ... p n", n=dstate)
        dstates = _chunk_scan_bwd_dstates(C, dA_cs, dout, dtype=states.dtype)
        dstates, ddA_chunk_cumsum, _dinit, states = _state_passing_bwd(
            rearrange(states, "... p n -> ... (p n)"), dA_cs[:, :, :, -1],
            rearrange(dstates, "... p n -> ... (p n)"), dfinal_states=None,
            has_initial_states=False, dstates_dtype=x.dtype, states_dtype=x.dtype,
            chunk_size=chunk_size)
        states = rearrange(states, "... (p n) -> ... p n", n=dstate)
        dstates = rearrange(dstates, "... (p n) -> ... p n", n=dstate)
        dx, ddt, _dD = _chunk_scan_chunk_state_bwd_dx(x, dt_hl, dA_cs, B, CB, dout, dstates,
                                                      D=None)
        dB, ddA_next = _chunk_state_bwd_db(x, dt_hl, dA_cs, dstates, B=B, ngroups=h)
        dC, ddA_cumsum_prev = _chunk_scan_bwd_dC(states.to(x.dtype), dA_cs, dout, C=C,
                                                 ngroups=h)
        dCB = _chunk_scan_bwd_dcb(x, dt_hl, dA_cs, dout, ngroups=h).to(CB.dtype)
        dB_given = torch.empty_like(B)
        dC_given = torch.empty_like(C)
        _bmm_chunk_bwd(C, dCB, residual=dB, out=dB_given)
        _bmm_chunk_bwd(B, rearrange(dCB, "... l s -> ... s l"), residual=dC, out=dC_given)
        # per-token d(alpha_log): the intra-chunk term + the chunk-state and cross-chunk terms
        # (upstream would feed this ddA into _chunk_cumsum_bwd to recouple with ddt — we don't)
        ddA_cumsum_prev[..., -1] += ddA_chunk_cumsum
        ddA_prev = ddA_cumsum_prev.flip([-1]).cumsum(dim=-1).flip([-1])
        ddA = _chunk_scan_bwd_ddAcs_stable(x, dt_hl, dA_cs, dout, CB) + ddA_next + ddA_prev
        dalpha_log = rearrange(ddA, "b h c s -> b (c s) h")
        ddt = rearrange(ddt, "b h c s -> b (c s) h")
        return dx, ddt, dalpha_log, dB_given, dC_given, None


def aum_ssd(x, dt, alpha_log, B, C, chunk_size=64):
    """The bare AUM SSD core (tests/benchmarks): x/B/C (b,l,h,d), dt/alpha_log (b,l,h)."""
    return _AumSSDCore.apply(x, dt, alpha_log, B, C, chunk_size)


def _build_operands(q, k, v, tau_bar, lam_bar, r, theta, dt_bias, eps, freqs):
    """The B2 operand pipeline: dynamics (fp32 — the decay/phase scalars are the numerically-
    sensitive piece), rotation ladder, L2 norms, write scale. Pure elementwise + cumsum, so it
    is torch.compile-fused on CUDA (measured 4.2x over eager at the reference shapes) instead
    of a hand-written kernel — same fusion win, no extra bug surface."""
    tau, alpha_log, rho, dphi = aum_dynamics(tau_bar.float(), lam_bar.float(), r.float(),
                                             theta.float(), dt_bias, eps)
    phi = torch.cumsum(dphi, dim=1)
    C = _rotate_ladder(q, phi.to(q.dtype), freqs)
    Bm = _rotate_ladder(_l2norm(k), phi.to(k.dtype), freqs)
    return C, Bm, _l2norm(v), rho * tau, alpha_log


_operands_impl = None


def _operands(*args):
    global _operands_impl
    if _operands_impl is None:
        try:
            _operands_impl = torch.compile(_build_operands, dynamic=False)
        except Exception:
            _operands_impl = _build_operands
    try:
        return _operands_impl(*args)
    except Exception:                       # inductor failure -> eager, permanently
        _operands_impl = _build_operands
        return _build_operands(*args)


def unfold_triton_chunk(q, k, v, tau_bar, lam_bar, r, theta, z=None, D=None, dt_bias=0.0,
                        eps=1e-4, chunk_size=64, norm_weight=None, freqs=None):
    """Drop-in for aum_unfold_chunk_ref's training path on CUDA (same args/semantics; the
    final (S_T, phi_T) capture is prefill-only and stays on the reference).

    Operands are built by the compiled fused elementwise pipeline, the core runs on the
    vendored upstream SSD kernels, the epilogue on the vendored gated-RMSNorm kernel."""
    B_, L, H, Dv = v.shape
    dtb = dt_bias.float() if torch.is_tensor(dt_bias) else torch.tensor(
        float(dt_bias), device=v.device)
    C, Bm, v_hat, dtw, alpha_log = _operands(q, k, v, tau_bar, lam_bar, r, theta, dtb, eps,
                                             freqs)
    Y = _AumSSDCore.apply(v_hat, dtw, alpha_log, Bm, C, chunk_size)
    if D is not None:
        Y = Y + (D if D.dim() == 2 else D.unsqueeze(-1)) * v
    if z is not None:
        w = (norm_weight if norm_weight is not None
             else torch.ones(Dv, device=Y.device, dtype=torch.float32)).repeat(H)
        Y = rmsnorm_fn(rearrange(Y, "b l h p -> (b l) (h p)"),
                       w, None, z=rearrange(z, "b l h p -> (b l) (h p)"),
                       eps=1e-5, group_size=Dv, norm_before_gate=True)
        Y = rearrange(Y, "(b l) (h p) -> b l h p", b=B_, p=Dv)
    return Y
