import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{Cosine}(x)=\cos(x)
\]
"""

@register_act_fn(name="cosine")
class Cosine(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.cos(x)

    def __repr__(self) -> str:
        return "Cosine()"
