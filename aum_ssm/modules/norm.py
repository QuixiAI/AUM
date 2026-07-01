# Pure-PyTorch RMSNorm (backend-agnostic; runs on CPU/MPS/CUDA without Triton).
# The Triton fused add+norm path (ops/triton/layer_norm.py) is a NVIDIA optimization
# used only when fused_add_norm is enabled; correctness lives here.
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x):
        out_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(out_dtype)
