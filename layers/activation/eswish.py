import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{ESwish}(x)=\beta x \cdot \sigma(x)
\]
"""

@register_act_fn(name="eswish")
class ESwish(nn.Module):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, x: Tensor) -> Tensor:
        return self.beta * x * torch.sigmoid(x)

    def __repr__(self) -> str:
        return f"ESwish(beta={self.beta})"
