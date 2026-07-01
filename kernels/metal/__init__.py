"""Self-contained PyTorch-MPS build for the AUM-Ø Metal kernels — no ThunderMittens repo needed.

Builds independently from source in THIS tree:
  - MSL substrate (tile primitives) is vendored in ./include,
  - the AUM kernels are in ./src (mamba2.metal SSD forward, mamba2_bwd.metal SSD backward),
  - ./aum_metal.mm is the torch-MPS dispatch (generic encoder + our two kernels' host ABI).

On import this compiles ./src/*.metal into aum.metallib with `xcrun metal` and JIT-builds the
ObjC++ dispatch via torch.utils.cpp_extension.load. Requirements: PyTorch (MPS) + Xcode's Metal
toolchain (`xcrun metal`). See NOTICE for attribution of the vendored substrate.
"""

import os
import subprocess

import torch  # noqa: F401  (extension links against torch)
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_INCLUDE = os.path.join(_HERE, "include")
_SRC = os.path.join(_HERE, "src")
_METALLIB = os.path.join(_HERE, "aum.metallib")
_METAL_SOURCES = [os.path.join(_SRC, "mamba2.metal"), os.path.join(_SRC, "mamba2_bwd.metal")]


def build_metallib(force: bool = False) -> str:
    """Compile ./src/*.metal into aum.metallib via `xcrun metal` (rebuild only if stale)."""
    if not force and os.path.exists(_METALLIB):
        if os.path.getmtime(_METALLIB) >= max(os.path.getmtime(s) for s in _METAL_SOURCES):
            return _METALLIB
    subprocess.run(["xcrun", "metal", "-std=metal3.1", "-O2", "-I", _INCLUDE,
                    *_METAL_SOURCES, "-o", _METALLIB], check=True)
    return _METALLIB


build_metallib()
_ext = load(
    name="aum_metal_ext",
    sources=[os.path.join(_HERE, "aum_metal.mm")],
    extra_cflags=["-std=c++17"],
    extra_ldflags=["-framework", "Metal", "-framework", "Foundation", "-framework", "QuartzCore"],
    verbose=False,
)
_ext._set_library(_METALLIB)


def mamba2(C, B, X, cumlog):
    """SSD forward ((C@Bᵀ)⊙exp(cl_i−cl_j)⊙causal)@X. C,B,X bf16 (B,H,N,D); cumlog fp32 (B,H,N).
    MPS; D in {64,128}, N%8."""
    return _ext.mamba2(C, B, X, cumlog)


def mamba2_bwd(C, B, X, cumlog, dY):
    """SSD backward -> (dC, dB, dX). C,B,X,dY bf16 (B,H,N,D); cumlog fp32 (B,H,N). MPS; D in {64,128}.
    dcumlog = <dY,Y> − <dX,X> is computed on the host by the caller (not returned)."""
    return _ext.mamba2_bwd(C, B, X, cumlog, dY)
