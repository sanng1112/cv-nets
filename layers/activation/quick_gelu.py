import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{QuickGELU}(x)=x\cdot\sigma(1.702x)
\]
"""

@register_act_fn(name="quick_gelu")
class QuickGELU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(1.702 * x)

    def __repr__(self) -> str:
        return "QuickGELU()"
