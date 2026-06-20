"""
cv-nets layers package.

Re-exports key builders from sub-packages for convenient access.
"""

from cvnets.layers.activation import (
    SUPPORTED_ACT_FNS as SUPPORTED_ACT_FNS,
    ACT_FN_MODULES as ACT_FN_MODULES,
    register_act_fn as register_act_fn,
    build_activation_layer as build_activation_layer,
    get_config_prop as get_config_prop,
)

from cvnets.layers.normalization import (
    SUPPORTED_NORM_FNS as SUPPORTED_NORM_FNS,
    NORM_LAYER_REGISTRY as NORM_LAYER_REGISTRY,
    register_norm_fn as register_norm_fn,
    build_normalization_layer as build_normalization_layer,
)

from cvnets.layers.pooling import (
    SUPPORTED_POOLING_LAYERS as SUPPORTED_POOLING_LAYERS,
    POOLING_LAYER_REGISTRY as POOLING_LAYER_REGISTRY,
    register_pooling_fn as register_pooling_fn,
    build_pooling_layer as build_pooling_layer,
)

from cvnets.layers.conv_layer import Conv2d
from cvnets.layers.drop_path import DropPath
from cvnets.layers.dropout import Dropout, Dropout2d
from cvnets.layers.flatten import Flatten
from cvnets.layers.layer_scale import LayerScale
from cvnets.layers.multi_head_attention import MultiHeadSelfAttention
from cvnets.layers.patch_embedding import PatchEmbedding
from cvnets.layers.positional_encoding import (
    sinusoidal_positional_encoding,
    LearnedPositionalEncoding,
)
from cvnets.layers.linear_layer import LinearLayer
from cvnets.layers.upsample import Upsample, ConvTranspose2d

__all__ = [
    "SUPPORTED_ACT_FNS",
    "ACT_FN_MODULES",
    "register_act_fn",
    "build_activation_layer",
    "get_config_prop",
    "SUPPORTED_NORM_FNS",
    "NORM_LAYER_REGISTRY",
    "register_norm_fn",
    "build_normalization_layer",
    "SUPPORTED_POOLING_LAYERS",
    "POOLING_LAYER_REGISTRY",
    "register_pooling_fn",
    "build_pooling_layer",
    "Conv2d",
    "DropPath",
    "Dropout",
    "Dropout2d",
    "Flatten",
    "LayerScale",
    "LinearLayer",
    "MultiHeadSelfAttention",
    "PatchEmbedding",
    "sinusoidal_positional_encoding",
    "LearnedPositionalEncoding",
    "Upsample",
    "ConvTranspose2d",
]
