import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{Swish}(x)=x\cdot \sigma(\beta x)
\]
"""

@register_act_fn(name="swish")
class Swish(nn.Module):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(self.beta * x)

    def __repr__(self) -> str:
        return f"Swish(beta={self.beta})"
