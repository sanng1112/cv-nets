import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{ACON}(x) = (p_1x-p_2x)\sigma(\beta(p_1x-p_2x))+p_2x
\]
"""

@register_act_fn(name="acon")
class ACON(nn.Module):
    def __init__(self, channels: int, use_channelwise_beta: bool = True) -> None:
        super().__init__()
        self.channels = channels
        shape = (1, channels, 1, 1) if use_channelwise_beta else (1, 1, 1, 1)
        self.p1 = nn.Parameter(torch.ones(shape))
        self.p2 = nn.Parameter(torch.zeros(shape))
        self.beta = nn.Parameter(torch.ones(shape))
        self.use_channelwise_beta = use_channelwise_beta

    def forward(self, x: Tensor) -> Tensor:
        d = self.p1 * x - self.p2 * x
        return d * torch.sigmoid(self.beta * d) + self.p2 * x

    def __repr__(self) -> str:
        return f"ACON(channels={self.channels}, use_channelwise_beta={self.use_channelwise_beta})"
