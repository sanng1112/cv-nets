import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{HardMish}(x)\approx \mathrm{Mish}(x)
\]
"""

@register_act_fn(name="hard_mish")
class HardMish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * torch.clamp(x + 2.0, min=0.0, max=2.0)

    def __repr__(self) -> str:
        return "HardMish()"
