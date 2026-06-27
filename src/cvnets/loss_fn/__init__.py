"""cv-nets loss function system.

Provides ``BaseLoss``, a ``register_loss_fn()`` decorator,
``build_loss_fn()`` factory, and the ``SUPPORTED_LOSSES`` list.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Type

from cvnets.core.registry import LOSS_REGISTRY
from cvnets.loss_fn.base_loss import BaseLoss

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

SUPPORTED_LOSSES: List[str] = []
LOSS_FN_MODULES: Dict[str, Type[BaseLoss]] = {}

_logger = logging.getLogger(__name__)


def register_loss_fn(name: str, category: str = ""):
    """Decorator: register *cls* under *name* in the local registry and
    the global ``LOSS_REGISTRY``.

    Parameters
    ----------
    name : str
        Loss function identifier.
    category : str
        Problem domain (e.g. ``'classification'``, ``'segmentation'``).
    """
    def decorator(cls):
        full_key = f"{category}/{name}" if category else name
        if full_key in SUPPORTED_LOSSES:
            raise ValueError(f"Cannot register duplicate loss function ({full_key})")
        SUPPORTED_LOSSES.append(full_key)
        LOSS_FN_MODULES[full_key] = cls
        LOSS_REGISTRY.register(name, category=category)(cls)
        return cls
    return decorator


def build_loss_fn(
    loss_type: str,
    category: str = "",
    *args,
    **kwargs,
) -> BaseLoss:
    """Construct and return a loss function.

    Parameters
    ----------
    loss_type : str
        Registered loss function identifier.
    category : str
        Problem domain.
    *args, **kwargs
        Forwarded to the loss constructor.

    Returns
    -------
    BaseLoss
        The constructed loss module.

    Raises
    ------
    ValueError
        If *loss_type* is not registered in *category*.
    """
    if not LOSS_REGISTRY.contains(loss_type, category=category):
        raise ValueError(
            f"Unknown loss function {loss_type!r} in category {category!r}. "
            f"Available: {SUPPORTED_LOSSES}"
        )
    return LOSS_REGISTRY.build(loss_type, category=category, *args, **kwargs)


# ---------------------------------------------------------------------------
# Import sub-packages so that @register_loss_fn decorators fire
# ---------------------------------------------------------------------------

from cvnets.loss_fn import classification as _  # noqa: F401
from cvnets.loss_fn import detection as _  # noqa: F401
from cvnets.loss_fn import metric_learning as _  # noqa: F401
from cvnets.loss_fn import regression as _  # noqa: F401
from cvnets.loss_fn import segmentation as _  # noqa: F401
from cvnets.loss_fn import ssl as _  # noqa: F401
