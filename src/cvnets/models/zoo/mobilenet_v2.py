"""
MobileNetV2 model zoo — InvertedResidual backbone.

Factory function ``make_mobilenet_v2()`` returns a plain ``nn.Module`` and is
registered in ``MODEL_REGISTRY``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch.nn as nn
from torch import Tensor

from cvnets.core.registry import MODEL_REGISTRY
from cvnets.blocks.inverted_residual import InvertedResidual
from cvnets.layers.conv_layer import Conv2d
from cvnets.layers.linear_layer import LinearLayer
from cvnets.layers.normalization import build_normalization_layer
from cvnets.utils.logger import info

# ---------------------------------------------------------------------------
# MobileNetV2 config: (expand_ratio, out_channels, num_blocks, stride)
# ---------------------------------------------------------------------------
# fmt: off
_MV2_CONFIG: List[Tuple[float, int, int, int]] = [
    (1, 16,  1, 1),
    (6, 24,  2, 2),
    (6, 32,  3, 2),
    (6, 64,  4, 2),
    (6, 96,  3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
]
# fmt: on


# ===================================================================
# _MobileNetV2
# ===================================================================


class _MobileNetV2(nn.Module):
    """MobileNetV2 backbone following Sandler et al. 2018.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 1000).
    width_mult : float
        Width multiplier applied to channel counts (default 1.0).
    """

    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        self.width_mult = width_mult

        def _adjust_channels(ch: int) -> int:
            """Apply width multiplier and round to nearest multiple of 8."""
            new_ch = int(ch * width_mult)
            return max(8, new_ch // 8 * 8) if new_ch % 8 != 0 else new_ch

        input_ch = _adjust_channels(32)

        # ---- Initial convolution ----
        features: list[nn.Module] = [
            nn.Sequential(
                Conv2d(3, input_ch, kernel_size=3, stride=2, padding=1, bias=False),
                build_normalization_layer(
                    {"type": "batch_norm", "num_features": input_ch},
                    num_features=input_ch,
                ),
                nn.ReLU6(inplace=True),
            )
        ]

        # ---- Inverted residual stages ----
        in_channels = input_ch
        for t, c, n, s in _MV2_CONFIG:
            out_channels = _adjust_channels(c)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        expand_ratio=t,
                    )
                )
                in_channels = out_channels

        # ---- Final 1×1 projection to 1280 ----
        last_channels = _adjust_channels(1280)
        features.append(
            nn.Sequential(
                Conv2d(
                    in_channels,
                    last_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                build_normalization_layer(
                    {"type": "batch_norm", "num_features": last_channels},
                    num_features=last_channels,
                ),
                nn.ReLU6(inplace=True),
            )
        )

        self.features = nn.Sequential(*features)

        # ---- Classifier head ----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LinearLayer(last_channels, num_classes, bias=True)

        # Log
        num_params = sum(p.numel() for p in self.parameters())
        info(f"_MobileNetV2 created — {num_params:,} parameters")

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# ===================================================================
# Factory function
# ===================================================================


@MODEL_REGISTRY.register("mobilenet_v2")
def make_mobilenet_v2(num_classes: int = 1000) -> _MobileNetV2:
    """Build a MobileNetV2 classification model.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 1000).

    Returns
    -------
    _MobileNetV2
    """
    return _MobileNetV2(num_classes=num_classes)
