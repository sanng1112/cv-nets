"""
ResNet-18/34 (ResidualBlock) and ResNet-50/101 (BottleneckBlock) model zoo.

All factory functions return a plain ``nn.Module`` (a ``_ResNet`` instance) and
are registered in ``MODEL_REGISTRY``.
"""

from __future__ import annotations

from typing import List, Tuple, Type

import torch.nn as nn
from torch import Tensor

from cvnets.core.registry import MODEL_REGISTRY
from cvnets.blocks.residual_block import ResidualBlock
from cvnets.blocks.bottleneck_block import BottleneckBlock
from cvnets.layers.linear_layer import LinearLayer
from cvnets.utils.logger import info

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_StageConfig = List[Tuple[int, int, int]]  # [(out_channels, num_blocks, stride), ...]

_RESNET_18_STAGES: _StageConfig = [
    (64, 2, 1),
    (128, 2, 2),
    (256, 2, 2),
    (512, 2, 2),
]

_RESNET_34_STAGES: _StageConfig = [
    (64, 3, 1),
    (128, 4, 2),
    (256, 6, 2),
    (512, 3, 2),
]

_RESNET_50_STAGES: _StageConfig = [
    (64, 3, 1),
    (128, 4, 2),
    (256, 6, 2),
    (512, 3, 2),
]

_RESNET_101_STAGES: _StageConfig = [
    (64, 3, 1),
    (128, 4, 2),
    (256, 23, 2),
    (512, 3, 2),
]


# ===================================================================
# _ResNet — generic ResNet backbone
# ===================================================================


class _ResNet(nn.Module):
    """Generic ResNet implementation.

    Parameters
    ----------
    block_cls : type
        Either ``ResidualBlock`` or ``BottleneckBlock``.
    stage_config : list of (out_channels, num_blocks, stride)
        Configuration for each of the four stages.
    num_classes : int
        Number of output classes (default 1000).
    expansion : int
        Channel expansion for the block (1 for ResidualBlock, 4 for BottleneckBlock).
    """

    def __init__(
        self,
        block_cls: Type[nn.Module],
        stage_config: _StageConfig,
        num_classes: int = 1000,
        expansion: int = 1,
    ) -> None:
        super().__init__()
        self.block_cls = block_cls
        self.expansion = expansion
        self.in_channels = 64

        # ---- Stem ----
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- Four stages ----
        self.layer1 = self._make_stage(stage_config[0])
        self.layer2 = self._make_stage(stage_config[1])
        self.layer3 = self._make_stage(stage_config[2])
        self.layer4 = self._make_stage(stage_config[3])

        # ---- Head ----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LinearLayer(self.in_channels, num_classes, bias=True)

        # Log param count
        num_params = sum(p.numel() for p in self.parameters())
        info(f"_ResNet({block_cls.__name__}) created — {num_params:,} parameters")

    def _make_stage(
        self, config: Tuple[int, int, int]
    ) -> nn.Sequential:
        """Build one stage of ``num_blocks`` residual blocks."""
        out_channels, num_blocks, stride = config
        blocks: list[nn.Module] = []

        # First block handles dimension change and stride
        blocks.append(
            self.block_cls(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
                expansion=self.expansion,
            )
        )
        self.in_channels = out_channels * self.expansion

        # Remaining blocks: same channels, stride=1
        for _ in range(1, num_blocks):
            blocks.append(
                self.block_cls(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1,
                    expansion=self.expansion,
                )
            )

        return nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# ===================================================================
# Factory functions
# ===================================================================


@MODEL_REGISTRY.register("resnet18")
def make_resnet18(num_classes: int = 1000) -> _ResNet:
    """Build a ResNet-18 with ``ResidualBlock`` (basic blocks).

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 1000).

    Returns
    -------
    _ResNet
    """
    return _ResNet(
        block_cls=ResidualBlock,
        stage_config=_RESNET_18_STAGES,
        num_classes=num_classes,
        expansion=1,
    )


@MODEL_REGISTRY.register("resnet34")
def make_resnet34(num_classes: int = 1000) -> _ResNet:
    """Build a ResNet-34 with ``ResidualBlock`` (basic blocks)."""
    return _ResNet(
        block_cls=ResidualBlock,
        stage_config=_RESNET_34_STAGES,
        num_classes=num_classes,
        expansion=1,
    )


@MODEL_REGISTRY.register("resnet50")
def make_resnet50(num_classes: int = 1000) -> _ResNet:
    """Build a ResNet-50 with ``BottleneckBlock`` (expansion=4)."""
    return _ResNet(
        block_cls=BottleneckBlock,
        stage_config=_RESNET_50_STAGES,
        num_classes=num_classes,
        expansion=4,
    )


@MODEL_REGISTRY.register("resnet101")
def make_resnet101(num_classes: int = 1000) -> _ResNet:
    """Build a ResNet-101 with ``BottleneckBlock`` (expansion=4)."""
    return _ResNet(
        block_cls=BottleneckBlock,
        stage_config=_RESNET_101_STAGES,
        num_classes=num_classes,
        expansion=4,
    )
