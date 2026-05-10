import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{Snake}(x)=x+\frac{1}{\alpha}\sin^2(\alpha x)
\]
"""

@register_act_fn(name="snake")
class Snake(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return x + (torch.sin(self.alpha * x) ** 2) / self.alpha

    def __repr__(self) -> str:
        return f"Snake(alpha={self.alpha})"
