"""
ResidualBlock (BasicBlock) — the fundamental building block of ResNet-style
architectures.

Structure::

    x → Conv2d(3×3) → BN → ReLU → Conv2d(3×3) → BN → + → ReLU → out
                                                          ↑
                                                          | (skip)
                                                      Conv2d(1×1)  [optional]
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cvnets.core.base_block import BaseBlock
from cvnets.core.registry import BLOCK_REGISTRY
from cvnets.layers.conv_layer import Conv2d
from cvnets.layers.normalization import build_normalization_layer


@BLOCK_REGISTRY.register("residual_basic")
@BLOCK_REGISTRY.register("ResidualBlock")
class ResidualBlock(BaseBlock):
    """
    A standard ResNet basic block with two 3×3 convolutions and a skip
    connection.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int
        Stride for the first convolution (default ``1``).
    use_norm : bool
        Whether to apply BatchNorm after each convolution (default ``True``).
    use_act : bool
        Whether to apply ReLU after each convolution (default ``True``).
    norm_cfg : dict or None
        Configuration for the normalisation layer (e.g.
        ``{"type": "batch_norm"}``).  If ``None``, defaults to batch norm.
    **kwargs
        Extra keyword arguments (accepted for config compatibility).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_norm: bool = True,
        use_act: bool = True,
        norm_cfg: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_norm = use_norm
        self.use_act = use_act

        # ---- Conv 1 ----
        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        if use_norm:
            norm_kwargs = {
                "type": norm_cfg.get("type", "batch_norm") if norm_cfg else "batch_norm",
                "num_features": out_channels,
            }
            self.norm1 = build_normalization_layer(opts=norm_kwargs, num_features=out_channels)
        else:
            self.norm1 = nn.Identity()

        # ---- Conv 2 ----
        self.conv2 = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        if use_norm:
            norm_kwargs2 = {
                "type": norm_cfg.get("type", "batch_norm") if norm_cfg else "batch_norm",
                "num_features": out_channels,
            }
            self.norm2 = build_normalization_layer(opts=norm_kwargs2, num_features=out_channels)
        else:
            self.norm2 = nn.Identity()

        # ---- Downsample (skip) ----
        self.downsample: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                (
                    build_normalization_layer(
                        opts={
                            "type": norm_cfg.get("type", "batch_norm") if norm_cfg else "batch_norm",
                            "num_features": out_channels,
                        },
                        num_features=out_channels,
                    )
                    if use_norm
                    else nn.Identity()
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.norm1(out)
        if self.use_act:
            out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.norm2(out)

        out += identity
        if self.use_act:
            out = F.relu(out, inplace=True)

        return out
