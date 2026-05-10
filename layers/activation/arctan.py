import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{Arctan}(x)=\arctan(x)
\]
"""

@register_act_fn(name="arctan")
class Arctan(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.atan(x)

    def __repr__(self) -> str:
        return "Arctan()"
