"""
Pooling layer system — lightweight implementation.

Provides a ``build_pooling_layer()`` factory that supports common
pooling types (avgpool, maxpool, adaptive_avg).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch.nn as nn

from cvnets.core.registry import POOLING_REGISTRY

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

SUPPORTED_POOLING_LAYERS: List[str] = []
POOLING_LAYER_REGISTRY: Dict[str, type] = {}


def register_pooling_fn(name: str):
    """Decorator: register *cls* under *name* in the local registry and
    the global ``POOLING_REGISTRY``.
    """
    def decorator(cls):
        if name in SUPPORTED_POOLING_LAYERS:
            raise ValueError(
                f"Cannot register duplicate pooling function ({name})"
            )
        SUPPORTED_POOLING_LAYERS.append(name)
        POOLING_LAYER_REGISTRY[name] = cls
        POOLING_REGISTRY.register(name)(cls)
        return cls
    return decorator


def build_pooling_layer(
    opts: Any = None,
    pool_type: Optional[str] = None,
    kernel_size: Optional[Any] = None,
    stride: Optional[Any] = None,
    padding: Optional[Any] = None,
    **kwargs: Any,
) -> Optional[nn.Module]:
    """Construct and return a pooling layer.

    Parameters
    ----------
    opts : dict or SimpleNamespace or None
        Configuration object (may contain ``type``, ``kernel_size``,
        ``stride``, ``padding``, ``output_size``, etc.).
    pool_type : str or None
        Explicit pooling type (overrides ``opts``).
    kernel_size : int or tuple or None
        Kernel size for the pooling operation.
    stride : int or tuple or None
        Stride for the pooling operation.
    padding : int or tuple or None
        Padding for the pooling operation.
    **kwargs
        Additional keyword arguments forwarded to the pooling class.

    Returns
    -------
    nn.Module or None
        The pooling layer, or ``None`` if no type is given.
    """
    if pool_type is None:
        if isinstance(opts, dict):
            pool_type = opts.get("type")
        else:
            pool_type = getattr(opts, "type", None) if opts is not None else None
    if not pool_type:
        return None

    pool_type = pool_type.lower()

    if pool_type in POOLING_LAYER_REGISTRY:
        pool_class = POOLING_LAYER_REGISTRY[pool_type]
        # Resolve kernel_size from opts if not passed directly
        if kernel_size is None and isinstance(opts, dict):
            kernel_size = opts.get("kernel_size")
        elif kernel_size is None and opts is not None:
            kernel_size = getattr(opts, "kernel_size", None)
        if stride is None and isinstance(opts, dict):
            stride = opts.get("stride")
        elif stride is None and opts is not None:
            stride = getattr(opts, "stride", None)
        if padding is None and isinstance(opts, dict):
            padding = opts.get("padding")
        elif padding is None and opts is not None:
            padding = getattr(opts, "padding", None)

        return pool_class(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )
    else:
        from cvnets.utils.logger import error as _log_error
        _log_error(
            f"Supported pooling layers: {SUPPORTED_POOLING_LAYERS}. "
            f"Supplied: {pool_type}"
        )
        raise NotImplementedError(
            f"Pooling type '{pool_type}' is not supported."
        )


# ---------------------------------------------------------------------------
# Built-in pooling classes
# ---------------------------------------------------------------------------


@register_pooling_fn("avgpool")
class AvgPool2d(nn.AvgPool2d):
    def __init__(
        self,
        kernel_size=None,
        stride=None,
        padding=None,
        **kwargs,
    ):
        _kernel_size = kernel_size if kernel_size is not None else 2
        _stride = stride if stride is not None else _kernel_size
        _padding = padding if padding is not None else 0
        super().__init__(
            kernel_size=_kernel_size,
            stride=_stride,
            padding=_padding,
        )


@register_pooling_fn("maxpool")
class MaxPool2d(nn.MaxPool2d):
    def __init__(
        self,
        kernel_size=None,
        stride=None,
        padding=None,
        **kwargs,
    ):
        _kernel_size = kernel_size if kernel_size is not None else 2
        _stride = stride if stride is not None else _kernel_size
        _padding = padding if padding is not None else 0
        super().__init__(
            kernel_size=_kernel_size,
            stride=_stride,
            padding=_padding,
        )


@register_pooling_fn("adaptive_avg")
class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size=None, **kwargs):
        _output_size = output_size if output_size is not None else 1
        super().__init__(output_size=_output_size)
