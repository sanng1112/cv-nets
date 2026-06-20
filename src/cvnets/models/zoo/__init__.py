"""
cv-nets model zoo — family of pre-configured architectures.

Each sub-module exposes factory functions registered in ``MODEL_REGISTRY``.
"""

from cvnets.models.zoo.resnet import (
    make_resnet18,
    make_resnet34,
    make_resnet50,
    make_resnet101,
)
from cvnets.models.zoo.mobilenet_v2 import make_mobilenet_v2
from cvnets.models.zoo.vit import make_vit_tiny, make_vit_small, make_vit_base
from cvnets.models.zoo.simple_cnn import simple_cnn

__all__ = [
    "make_resnet18",
    "make_resnet34",
    "make_resnet50",
    "make_resnet101",
    "make_mobilenet_v2",
    "make_vit_tiny",
    "make_vit_small",
    "make_vit_base",
    "simple_cnn",
]
