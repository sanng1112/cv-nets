import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import register_act_fn


r"""
\[
\mathrm{MetaACON}(x) = (p_1x-p_2x)\sigma(\beta(x)(p_1x-p_2x))+p_2x
\]
"""

@register_act_fn(name="meta_acon")
class MetaACON(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channels = channels
        hidden = max(channels // reduction, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)
        self.p1 = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.p2 = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        beta = torch.sigmoid(self.fc2(F.relu(self.fc1(self.avgpool(x)), inplace=True)))
        d = self.p1 * x - self.p2 * x
        return d * torch.sigmoid(beta * d) + self.p2 * x

    def __repr__(self) -> str:
        return f"MetaACON(channels={self.channels}, reduction={self.fc1.out_channels})"
