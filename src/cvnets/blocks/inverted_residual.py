"""
InvertedResidual — the core building block of MobileNetV2 (Sandler et al., 2018).

Structure::

    x → Conv2d(1×1, expand) → BN → Act → DWConv(3×3) → BN → Act
        → Conv2d(1×1, project) → BN → + (skip when stride==1 and
                                          in_channels==out_channels)

The expansion activation can use ReLU6 (default) or Hardswish.
The projection (final 1×1) has *no* activation.
"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn.functional as F
from torch import Tensor, nn

from cvnets.core.base_block import BaseBlock
from cvnets.core.registry import BLOCK_REGISTRY
from cvnets.layers.conv_layer import Conv2d
from cvnets.layers.normalization import build_normalization_layer
from cvnets.utils.logger import info


@BLOCK_REGISTRY.register("inverted_residual")
@BLOCK_REGISTRY.register("InvertedResidual")
class InvertedResidual(BaseBlock):
    """
    Inverted residual block from MobileNetV2.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int
        Stride for the depthwise convolution (must be ``1`` or ``2``).
    expand_ratio : float
        Expansion factor applied to ``in_channels`` to compute the hidden
        dimension (default ``6.0``, i.e. 6× expansion).
    use_hardswish : bool
        If ``True``, use ``Hardswish`` activation for the expansion;
        otherwise use ``ReLU6`` (default ``False``).
    use_norm : bool
        Whether to apply BatchNorm after each convolution (default ``True``).
    norm_cfg : dict or None
        Configuration for the normalisation layer.
    **kwargs
        Extra keyword arguments (accepted for config compatibility).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: float = 6.0,
        use_hardswish: bool = False,
        use_norm: bool = True,
        norm_cfg: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.use_hardswish = use_hardswish

        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: list = []

        # ---- Pointwise expansion (1×1) ----
        if expand_ratio != 1.0:
            layers.append(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            )
            if use_norm:
                norm_kwargs = {
                    "type": norm_cfg.get("type", "batch_norm") if norm_cfg else "batch_norm",
                    "num_features": hidden_dim,
                }
                layers.append(
                    build_normalization_layer(opts=norm_kwargs, num_features=hidden_dim)
                )
            layers.append(self._build_act())
        else:
            # When expand_ratio == 1, the expansion conv is skipped
            hidden_dim = in_channels

        # ---- Depthwise (3×3) ----
        layers.append(
            Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False,
            )
        )
        if use_norm:
            norm_kwargs = {
                "type": norm_cfg.get("type", "batch_norm") if norm_cfg else "batch_norm",
                "num_features": hidden_dim,
            }
            layers.append(
                build_normalization_layer(opts=norm_kwargs, num_features=hidden_dim)
            )
        layers.append(self._build_act())

        # ---- Pointwise projection (1×1, no activation) ----
        layers.append(
            Conv2d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        )
        if use_norm:
            norm_kwargs = {
                "type": norm_cfg.get("type", "batch_norm") if norm_cfg else "batch_norm",
                "num_features": out_channels,
            }
            layers.append(
                build_normalization_layer(opts=norm_kwargs, num_features=out_channels)
            )

        self.block = nn.Sequential(*layers)

        if self.use_residual:
            info(
                f"InvertedResidual(in={in_channels}, out={out_channels}, "
                f"stride={stride}, expand={expand_ratio}): residual ON"
            )
        else:
            info(
                f"InvertedResidual(in={in_channels}, out={out_channels}, "
                f"stride={stride}, expand={expand_ratio}): residual OFF"
            )

    def _build_act(self) -> nn.Module:
        """Return the activation module for the expansion / depthwise stage."""
        if self.use_hardswish:
            return nn.Hardswish(inplace=True)
        return nn.ReLU6(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)
