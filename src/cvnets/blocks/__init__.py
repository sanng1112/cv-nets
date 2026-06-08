"""
cv-nets blocks package.
"""

from cvnets.blocks.registry import build_block
from cvnets.blocks.conv_bn_act import ConvBNAct
from cvnets.blocks.se_block import SEBlock
from cvnets.blocks.mlp_block import TransformerMLP
from cvnets.blocks.transformer_block import TransformerEncoderBlock
from cvnets.blocks.depthwise_separable_conv import DepthwiseSeparableConvBlock

__all__ = [
    "build_block",
    "ConvBNAct",
    "SEBlock",
    "TransformerMLP",
    "TransformerEncoderBlock",
    "DepthwiseSeparableConvBlock",
]
