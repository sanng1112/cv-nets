import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import register_act_fn


r"""
\[
\mathrm{ReGLU}(x)=a\odot \mathrm{ReLU}(b), \quad [a,b]=\mathrm{chunk}(x)
\]
"""

@register_act_fn(name="reglu")
class ReGLU(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        a, b = torch.chunk(x, 2, dim=self.dim)
        return a * F.relu(b)

    def __repr__(self) -> str:
        return f"ReGLU(dim={self.dim})"
