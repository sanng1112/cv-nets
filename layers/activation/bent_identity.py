import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{BentIdentity}(x)=\frac{\sqrt{x^2+1}-1}{2}+x
\]
"""

@register_act_fn(name="bent_identity")
class BentIdentity(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return ((torch.sqrt(x * x + 1.0) - 1.0) / 2.0) + x

    def __repr__(self) -> str:
        return "BentIdentity()"
