"""Triton/CUDA twins of the AUM-Ø Metal kernels (kernels/metal is the SPEC).

Same buffer ABI as kernels.metal — pack layouts are defined once in
aum_ssm/ops/metal/silence_metal.py and hardcoded identically here — so the host plumbing
(pack builders, autograd.Function, backward GEMM assembly) is shared between backends.
"""

from kernels.triton.silence import aum_silence_fwd, aum_silence_bwd  # noqa: F401
