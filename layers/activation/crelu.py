import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import register_act_fn


r"""
\[
\mathrm{CReLU}(x)=\big[\mathrm{ReLU}(x),\mathrm{ReLU}(-x)\big]
\]
"""

@register_act_fn(name="crelu")
class CReLU(nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([F.relu(x), F.relu(-x)], dim=self.dim)

    def __repr__(self) -> str:
        return f"CReLU(dim={self.dim})"
