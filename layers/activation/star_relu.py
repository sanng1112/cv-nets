import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import register_act_fn


r"""
\[
\mathrm{StarReLU}(x)=s\cdot \mathrm{ReLU}(x)^2+b
\]
"""

@register_act_fn(name="star_relu")
class StarReLU(nn.Module):
    def __init__(self, scale: float = 1.0, bias: float = 0.0, inplace: bool = False) -> None:
        super().__init__()
        self.scale = scale
        self.bias = bias
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        y = F.relu(x, inplace=self.inplace)
        return self.scale * (y * y) + self.bias

    def __repr__(self) -> str:
        return f"StarReLU(scale={self.scale}, bias={self.bias}, inplace={self.inplace})"
