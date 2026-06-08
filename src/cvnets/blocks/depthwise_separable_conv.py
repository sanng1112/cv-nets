"""Depthwise Separable Convolution — efficient conv block (Howard et al., 2017)."""

from typing import Any

from torch import Tensor, nn

from cvnets.core.base_block import BaseBlock
from cvnets.core.registry import BLOCK_REGISTRY


@BLOCK_REGISTRY.register("depthwise_separable_conv")
@BLOCK_REGISTRY.register("DepthwiseSeparableConvBlock")
class DepthwiseSeparableConvBlock(BaseBlock):
    """Depthwise separable convolution block.

    Factorises a standard convolution into a depthwise convolution
    (one filter per input channel) followed by a pointwise (1×1)
    convolution.  Significantly reduces parameters and FLOPs.

    Structure: ``DWConv → [Norm] → [Act] → PWConv → [Norm] → [Act]``

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Kernel size for the depthwise convolution (default ``3``).
    stride : int
        Stride (default ``1``).
    padding : int or None
        Padding.  If ``None``, computed as ``kernel_size // 2``.
    use_norm : bool
        Whether to apply BatchNorm after each convolution (default ``True``).
    use_act : bool
        Whether to apply ReLU6 after each convolution (default ``True``).
    **kwargs
        Extra keyword arguments (ignored; accepted for config compatibility).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        use_norm: bool = True,
        use_act: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        layers: list = []

        # Depthwise convolution
        layers.append(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            )
        )
        if use_norm:
            layers.append(nn.BatchNorm2d(in_channels))
        if use_act:
            layers.append(nn.ReLU6(inplace=True))

        # Pointwise convolution
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_act:
            layers.append(nn.ReLU6(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.block[0].in_channels}, "
            f"out_channels={self.block[-3].out_channels}"
        )
