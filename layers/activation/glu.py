import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{GLU}(x)=a\odot\sigma(b), \quad [a,b]=\mathrm{chunk}(x)
\]
"""

@register_act_fn(name="glu")
class GLU(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        a, b = torch.chunk(x, 2, dim=self.dim)
        return a * torch.sigmoid(b)

    def __repr__(self) -> str:
        return f"GLU(dim={self.dim})"
