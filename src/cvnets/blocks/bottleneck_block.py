"""
BottleneckBlock — deeper residual block with 1×1 → 3×3 → 1×1 structure.

Used in ResNet-50/101/152 architectures.  The three convolutions reduce
then expand the channel dimension, making the block computationally cheaper
than a plain 3×3→3×3 block of the same width.

Structure::

    x → Conv2d(1×1, reduce) → BN → ReLU → Conv2d(3×3) → BN → ReLU
        → Conv2d(1×1, expand) → BN → + → ReLU → out
                                   ↑
                               Conv2d(1×1)  [optional downsample]
"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn.functional as F
from torch import Tensor, nn

from cvnets.core.base_block import BaseBlock
from cvnets.core.registry import BLOCK_REGISTRY
from cvnets.layers.conv_layer import Conv2d
from cvnets.layers.normalization import build_normalization_layer


@BLOCK_REGISTRY.register("bottleneck")
@BLOCK_REGISTRY.register("BottleneckBlock")
class BottleneckBlock(BaseBlock):
    """
    Bottleneck residual block with 1×1 → 3×3 → 1×1 convs, BN + ReLU after
    each, and an optional identity downsample.

    The expansion ratio controls the output channel dimension::

        out_channels = in_channels * expansion

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (before expansion).
    expansion : int
        Channel multiplier for the final 1×1 conv (default ``4``, matching
        ResNet-50/101/152).
    stride : int
        Stride for the 3×3 convolution (default ``1``).
    use_norm : bool
        Whether to apply BatchNorm after each convolution (default ``True``).
    use_act : bool
        Whether to apply ReLU after each convolution (default ``True``).
    norm_cfg : dict or None
        Configuration for the normalisation layer.
    **kwargs
        Extra keyword arguments (accepted for config compatibility).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 4,
        stride: int = 1,
        use_norm: bool = True,
        use_act: bool = True,
        norm_cfg: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.stride = stride
        self.use_norm = use_norm
        self.use_act = use_act

        hidden_dim = out_channels
        expanded_dim = out_channels * expansion

        # ---- 1×1 reduce ----
        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.norm1 = self._build_norm(hidden_dim, norm_cfg)

        # ---- 3×3 ----
        self.conv2 = Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm2 = self._build_norm(hidden_dim, norm_cfg)

        # ---- 1×1 expand ----
        self.conv3 = Conv2d(
            in_channels=hidden_dim,
            out_channels=expanded_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.norm3 = self._build_norm(expanded_dim, norm_cfg)

        # ---- Downsample (skip) ----
        self.downsample: nn.Module = nn.Identity()
        if stride != 1 or in_channels != expanded_dim:
            layers: list = []
            layers.append(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=expanded_dim,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                )
            )
            if use_norm:
                layers.append(self._build_norm(expanded_dim, norm_cfg))
            self.downsample = nn.Sequential(*layers)

    def _build_norm(self, num_features: int, norm_cfg: Optional[dict]) -> nn.Module:
        """Build a normalisation layer or return Identity."""
        if not self.use_norm:
            return nn.Identity()
        cfg = {
            "type": norm_cfg.get("type", "batch_norm") if norm_cfg else "batch_norm",
            "num_features": num_features,
        }
        return build_normalization_layer(opts=cfg, num_features=num_features)

    def _apply_act(self, x: Tensor) -> Tensor:
        """Apply ReLU if activation is enabled."""
        if self.use_act:
            return F.relu(x, inplace=True)
        return x

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self._apply_act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self._apply_act(out)

        out = self.conv3(out)
        out = self.norm3(out)

        out += identity
        out = self._apply_act(out)

        return out
