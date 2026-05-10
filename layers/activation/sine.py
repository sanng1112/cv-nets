import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{Sine}(x)=\sin(x)
\]
"""

@register_act_fn(name="sine")
class Sine(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)

    def __repr__(self) -> str:
        return "Sine()"
