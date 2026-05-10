import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{FReLU}(x)=\max(x,\mathrm{BN}(\mathrm{DWConv}(x)))
\]
"""

@register_act_fn(name="frelu")
class FReLU(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.dw_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.channels = channels
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        return torch.max(x, self.bn(self.dw_conv(x)))

    def __repr__(self) -> str:
        return f"FReLU(channels={self.channels}, kernel_size={self.kernel_size})"
