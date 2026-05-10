import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import register_act_fn


r"""
\[
y = a(x)\,\mathrm{ReLU}(x) - b(x)\,\mathrm{ReLU}(-x)
\]
"""

@register_act_fn(name="dynamic_relu")
class DynamicReLU(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels * 2, kernel_size=1, bias=True)
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        g = self.avgpool(x)
        g = F.relu(self.fc1(g), inplace=True)
        g = self.fc2(g)
        a, b = torch.chunk(g, 2, dim=1)
        a = torch.sigmoid(a)
        b = torch.sigmoid(b)
        return a * F.relu(x) - b * F.relu(-x)

    def __repr__(self) -> str:
        return f"DynamicReLU(channels={self.channels}, reduction={self.fc1.out_channels})"
