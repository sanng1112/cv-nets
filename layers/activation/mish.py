import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import register_act_fn


r"""
\[
\mathrm{Mish}(x)=x\tanh(\log(1+e^x))
\]
"""

@register_act_fn(name="mish")
class Mish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(F.softplus(x))

    def __repr__(self) -> str:
        return "Mish()"
