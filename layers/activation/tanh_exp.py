import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{TanhExp}(x)=x\tanh(e^x)
\]
"""

@register_act_fn(name="tanh_exp")
class TanhExp(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(torch.exp(x))

    def __repr__(self) -> str:
        return "TanhExp()"
