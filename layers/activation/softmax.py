import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import register_act_fn


r"""
\[
\mathrm{Softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}
\]
"""

@register_act_fn(name="softmax")
class Softmax(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x, dim=self.dim)

    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"
