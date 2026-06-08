"""
Normalization layer system — lightweight implementation.

Provides a ``build_normalization_layer()`` factory that supports common
normalization types (batch_norm, layer_norm, group_norm, instance_norm).
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

import torch.nn as nn

from cvnets.core.registry import NORMALIZATION_REGISTRY

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

SUPPORTED_NORM_FNS: list = []
NORM_LAYER_REGISTRY: Dict[str, type] = {}


def register_norm_fn(name: str):
    """Decorator: register *cls* under *name* in the local registry and
    the global ``NORMALIZATION_REGISTRY``.
    """
    def decorator(cls):
        if name in SUPPORTED_NORM_FNS:
            raise ValueError(
                f"Cannot register duplicate normalization function ({name})"
            )
        SUPPORTED_NORM_FNS.append(name)
        NORM_LAYER_REGISTRY[name] = cls
        NORMALIZATION_REGISTRY.register(name)(cls)
        return cls
    return decorator


def build_normalization_layer(
    opts: Any = None,
    num_features: Optional[int] = None,
    norm_type: Optional[str] = None,
    num_groups: Optional[int] = None,
    momentum: Optional[float] = None,
) -> nn.Module:
    """Construct and return a normalisation layer.

    Parameters
    ----------
    opts : dict or SimpleNamespace or None
        Configuration object (may contain ``type``, ``num_features``,
        ``groups``, ``momentum``).
    num_features : int or None
        Number of channels / features.
    norm_type : str or None
        Explicit normalisation type (overrides ``opts``).
    num_groups : int or None
        Number of groups for group normalisation.
    momentum : float or None
        Momentum for batch norm.

    Returns
    -------
    nn.Module
        The normalisation layer (``Identity`` if ``norm_type`` is falsy).
    """
    if norm_type is None:
        if isinstance(opts, dict):
            norm_type = opts.get("type")
        else:
            norm_type = getattr(opts, "type", None) if opts is not None else None
    if not norm_type:
        return nn.Identity()

    if num_features is None:
        if isinstance(opts, dict):
            num_features = opts.get("num_features")
        else:
            num_features = getattr(opts, "num_features", None)
    if num_features is None:
        return nn.Identity()

    if num_groups is None:
        if isinstance(opts, dict):
            num_groups = opts.get("groups", 1)
        else:
            num_groups = getattr(opts, "groups", 1) if opts is not None else 1
    if momentum is None:
        if isinstance(opts, dict):
            momentum = opts.get("momentum", 0.1)
        else:
            momentum = getattr(opts, "momentum", 0.1) if opts is not None else 0.1

    norm_type = norm_type.lower()

    if norm_type in NORM_LAYER_REGISTRY:
        norm_class = NORM_LAYER_REGISTRY[norm_type]
        init_sig = inspect.signature(norm_class.__init__)
        build_kwargs: Dict[str, Any] = {}
        if "num_features" in init_sig.parameters:
            build_kwargs["num_features"] = num_features
        if "normalized_shape" in init_sig.parameters:
            build_kwargs["normalized_shape"] = num_features
        if "num_groups" in init_sig.parameters:
            build_kwargs["num_groups"] = num_groups
        if "momentum" in init_sig.parameters:
            build_kwargs["momentum"] = momentum
        return norm_class(**build_kwargs)
    elif norm_type == "identity":
        return nn.Identity()
    else:
        from cvnets.utils.logger import error as _log_error
        _log_error(
            f"Supported normalisation layers: {SUPPORTED_NORM_FNS}. "
            f"Got: {norm_type}"
        )
        raise NotImplementedError(
            f"Normalisation type '{norm_type}' is not supported."
        )


# Register built-in normalisation layers
@register_norm_fn("batch_norm")
class _BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, **kwargs):
        super().__init__(num_features, **kwargs)


@register_norm_fn("layer_norm")
class _LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, **kwargs):
        super().__init__(normalized_shape, **kwargs)


@register_norm_fn("group_norm")
class _GroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_features, **kwargs):
        super().__init__(num_groups, num_features, **kwargs)


@register_norm_fn("instance_norm")
class _InstanceNorm2d(nn.InstanceNorm2d):
    def __init__(self, num_features, **kwargs):
        super().__init__(num_features, **kwargs)


from cvnets.layers.normalization.rms_norm import RMSNorm as _RMSNorm
register_norm_fn("rms_norm")(_RMSNorm)
