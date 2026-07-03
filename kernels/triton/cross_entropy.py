# Fused cross-entropy fwd/bwd over the vocab axis — Triton/CUDA twin of
# kernels/metal/src/cross_entropy.metal (the SPEC). One program per row, looping the vocab dim
# in BV chunks with an online (max, sumexp) state. Never stores the (T, V) probabilities: fwd
# emits per-row loss + lse (natural-log domain), bwd recomputes p = exp(x - lse) on the fly.
# Supports ignore_index (masked rows -> 0 loss / 0 grad), label smoothing, a z-loss regularizer
# (z_loss * lse^2), and the Gemma-2 softcap (z -> softcap * tanh(z / softcap)).

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_NEG = tl.constexpr(-3.4028234663852886e38)   # finite -FLT_MAX (matches Metal; avoids inf-inf)


@triton.jit
def _softcap(x, softcap, HAS_CAP: tl.constexpr):
    if HAS_CAP:
        return softcap * libdevice.tanh(x / softcap)
    else:
        return x


@triton.jit
def _ce_fwd_kernel(logits, targets, loss, lse_out, V, ignore_index,
                   label_smoothing, z_loss, softcap,
                   BV: tl.constexpr, HAS_SMOOTH: tl.constexpr, HAS_Z: tl.constexpr,
                   HAS_CAP: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = logits + row * V
    y = tl.load(targets + row)
    if y == ignore_index:
        tl.store(loss + row, 0.0)
        tl.store(lse_out + row, 0.0)
    else:
        m = tl.full([BV], _NEG, dtype=tl.float32)
        l = tl.zeros([BV], dtype=tl.float32)
        sx = 0.0
        for v0 in range(0, V, BV):
            idx = v0 + tl.arange(0, BV)
            mask = idx < V
            x = _softcap(tl.load(base + idx, mask=mask, other=_NEG).to(tl.float32),
                         softcap, HAS_CAP)
            nm = tl.maximum(m, x)
            l = l * tl.exp(m - nm) + tl.where(mask, tl.exp(x - nm), 0.0)
            m = nm
            if HAS_SMOOTH:
                sx += tl.sum(tl.where(mask, x, 0.0))
        M = tl.max(m)
        lse = M + tl.log(tl.sum(l * tl.exp(m - M)))
        x_y = _softcap(tl.load(base + y).to(tl.float32), softcap, HAS_CAP)
        ls = (1.0 - label_smoothing) * (lse - x_y)
        if HAS_SMOOTH:
            ls += label_smoothing * (lse - sx / V)
        if HAS_Z:
            ls += z_loss * lse * lse
        tl.store(loss + row, ls)
        tl.store(lse_out + row, lse)


@triton.jit
def _ce_bwd_kernel(logits, targets, lse_in, grad_out, grad_logits, V, ignore_index,
                   label_smoothing, z_loss, softcap,
                   BV: tl.constexpr, HAS_SMOOTH: tl.constexpr, HAS_Z: tl.constexpr,
                   HAS_CAP: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = logits + row * V
    gbase = grad_logits + row * V
    y = tl.load(targets + row)
    if y == ignore_index:
        for v0 in range(0, V, BV):
            idx = v0 + tl.arange(0, BV)
            tl.store(gbase + idx, tl.zeros([BV], dtype=tl.float32), mask=idx < V)
    else:
        lse = tl.load(lse_in + row)
        go = tl.load(grad_out + row)
        zc = 1.0 + 2.0 * z_loss * lse if HAS_Z else 1.0
        smooth = label_smoothing / V
        for v0 in range(0, V, BV):
            idx = v0 + tl.arange(0, BV)
            mask = idx < V
            capped = _softcap(tl.load(base + idx, mask=mask, other=_NEG).to(tl.float32),
                              softcap, HAS_CAP)
            p = tl.exp(capped - lse)
            g = zc * p - smooth - (1.0 - label_smoothing) * tl.where(idx == y, 1.0, 0.0)
            if HAS_CAP:
                t = capped / softcap        # grad through tanh: 1 - (capped/softcap)^2
                g *= 1.0 - t * t
            tl.store(gbase + idx, g * go, mask=mask)


def cross_entropy_fwd(logits, targets, ignore_index=-100, label_smoothing=0.0, z_loss=0.0,
                      softcap=0.0):
    """Same contract as kernels.metal.cross_entropy_fwd: (loss (T,), lse (T,)) fp32;
    logits (T, V) f32/f16/bf16 contiguous."""
    T, V = logits.shape
    logits = logits.contiguous()
    loss = torch.empty(T, dtype=torch.float32, device=logits.device)
    lse = torch.empty(T, dtype=torch.float32, device=logits.device)
    _ce_fwd_kernel[(T,)](logits, targets.contiguous().int(), loss, lse, V,
                         int(ignore_index), float(label_smoothing), float(z_loss),
                         float(softcap), BV=2048, HAS_SMOOTH=label_smoothing > 0,
                         HAS_Z=z_loss > 0, HAS_CAP=softcap > 0, num_warps=4)
    return loss, lse


def cross_entropy_bwd(logits, targets, lse, grad_out, ignore_index=-100, label_smoothing=0.0,
                      z_loss=0.0, softcap=0.0):
    """Same contract as kernels.metal.cross_entropy_bwd: grad_logits (T, V) in logits.dtype,
    scaled by per-row grad_out."""
    T, V = logits.shape
    logits = logits.contiguous()
    grad = torch.empty_like(logits)
    _ce_bwd_kernel[(T,)](logits, targets.contiguous().int(), lse, grad_out.contiguous(),
                         grad, V, int(ignore_index), float(label_smoothing), float(z_loss),
                         float(softcap), BV=2048, HAS_SMOOTH=label_smoothing > 0,
                         HAS_Z=z_loss > 0, HAS_CAP=softcap > 0, num_warps=4)
    return grad
