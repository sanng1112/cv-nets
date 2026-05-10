import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{LiSHT}(x)=x\tanh(x)
\]
"""

@register_act_fn(name="lisht")
class LiSHT(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(x)

    def __repr__(self) -> str:
        return "LiSHT()"
