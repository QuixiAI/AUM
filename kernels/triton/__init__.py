"""Triton/CUDA twins of the AUM-Ø Metal kernels (kernels/metal is the SPEC).

Same buffer ABI as kernels.metal — pack layouts are defined once in
aum_ssm/ops/metal/silence_metal.py and hardcoded identically here — so the host plumbing
(pack builders, autograd.Function, backward GEMM assembly) is shared between backends.
"""

from kernels.triton.silence import aum_silence_fwd, aum_silence_bwd  # noqa: F401
from kernels.triton.cross_entropy import cross_entropy_fwd, cross_entropy_bwd  # noqa: F401


def fused_linear_cross_entropy_ce(h, W, targets, row_weight=None, divisor=None,
                                  chunk_rows=2048, ignore_index=-100):
    """Liger-style chunked fused-linear-CE core (same contract as the kernels.metal twin):
    per-row CE of (h @ W.T) vs targets WITHOUT ever materializing the full (T,V) logits, with
    optional per-row loss weights (the §8 mixture w).

    h (T,K), W (V,K), targets (T,), row_weight (T,) or None; divisor defaults to the number of
    non-ignored rows. Returns (loss = sum(row_weight*ce)/divisor, dh (T,K), dW (V,K), ce (T,))
    — grads w.r.t. h and W for that weighted mean; the grad w.r.t. row_weight is ce/divisor.
    """
    import torch as _t
    T = h.shape[0]
    if divisor is None:
        divisor = int((targets != ignore_index).sum().clamp(min=1).item())
    dh = _t.zeros_like(h)
    dW = _t.zeros_like(W)
    ce_all = _t.empty(T, dtype=_t.float32, device=h.device)
    total = _t.zeros((), dtype=_t.float32, device=h.device)
    for c0 in range(0, T, chunk_rows):
        c1 = min(T, c0 + chunk_rows)
        hc, tc = h[c0:c1], targets[c0:c1]
        logits = hc @ W.T
        ce, lse_c = cross_entropy_fwd(logits, tc, ignore_index)
        ce_all[c0:c1] = ce
        wc = row_weight[c0:c1].float() if row_weight is not None else None
        total += (ce * wc).sum() if wc is not None else ce.sum()
        go = (wc / divisor) if wc is not None \
            else _t.full((c1 - c0,), 1.0 / divisor, device=h.device)
        g = cross_entropy_bwd(logits, tc, lse_c, go, ignore_index)
        dh[c0:c1] = (g @ W).to(h.dtype)
        dW += (g.T @ hc).to(W.dtype)
    return total / divisor, dh, dW, ce_all
