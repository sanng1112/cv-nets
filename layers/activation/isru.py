import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{ISRU}(x)=\frac{x}{\sqrt{1+\alpha x^2}}
\]
"""

@register_act_fn(name="isru")
class ISRU(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return x / torch.sqrt(1.0 + self.alpha * x * x)

    def __repr__(self) -> str:
        return f"ISRU(alpha={self.alpha})"
