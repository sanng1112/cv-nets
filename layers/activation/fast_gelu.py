import math
import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{FastGELU}(x)=\frac{1}{2}x\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}(x+0.044715x^3)\right)\right)
\]
"""

@register_act_fn(name="fast_gelu")
class FastGELU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

    def __repr__(self) -> str:
        return "FastGELU()"
