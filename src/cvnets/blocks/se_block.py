"""Squeeze-and-Excitation — channel-wise attention gating (Hu et al., 2018)."""

from typing import Any

from torch import Tensor, nn

from cvnets.core.base_block import BaseBlock
from cvnets.core.registry import BLOCK_REGISTRY


@BLOCK_REGISTRY.register("se_block")
@BLOCK_REGISTRY.register("SEBlock")
class SEBlock(BaseBlock):
    """Squeeze-and-Excitation channel attention block.

    Applies global average pooling → two-layer FC → sigmoid to produce
    per-channel excitation weights that recalibrate channel-wise feature
    responses.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    reduction : int
        Reduction ratio for the bottleneck FC layer (default ``16``).
    **kwargs
        Extra keyword arguments (ignored; accepted for config compatibility).
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        bottleneck = max(1, in_channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, _, _ = x.shape
        y = self.pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y

    def extra_repr(self) -> str:
        in_ch = self.fc[0].in_features
        bottleneck = self.fc[0].out_features
        return f"in_channels={in_ch}, bottleneck={bottleneck}"
