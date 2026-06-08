# NOTE: torch-dependent base modules are imported lazily to allow
# standalone use of non-torch components (e.g. ConfigResolver,
# Registry).  Direct access via ``from cvnets.core.base_layer import BaseLayer``
# still works when torch is available.

from cvnets.core.exceptions import (
    ConfigError,
    LayerDefinitionError,
    ModelBuildError,
)

__all__ = [
    "BaseLayer",
    "BaseBlock",
    "BaseModel",
    "ConfigError",
    "LayerDefinitionError",
    "ModelBuildError",
]


def __getattr__(name: str):
    """Lazily expose torch-dependent base classes."""
    import importlib

    _LAZY_MAP = {
        "BaseLayer": "cvnets.core.base_layer",
        "BaseBlock": "cvnets.core.base_block",
        "BaseModel": "cvnets.core.base_model",
    }
    if name in _LAZY_MAP:
        module = importlib.import_module(_LAZY_MAP[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

