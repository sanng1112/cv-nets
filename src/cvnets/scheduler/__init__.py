"""
Scheduler package for cv-nets.

Provides wrapper classes for PyTorch LR schedulers that can be registered
and built via the ``SCHED_REGISTRY``. The main entry point is
``build_scheduler()``.
"""
from __future__ import annotations

from typing import Any

from torch import nn
from torch.optim.optimizer import Optimizer

from cvnets.scheduler.registry import SCHED_REGISTRY, register_scheduler
from cvnets.scheduler.cosine import CosineAnnealingLRWrapper
from cvnets.scheduler.step import StepLRWrapper
from cvnets.scheduler.one_cycle import OneCycleLRWrapper

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SUPPORTED_SCHEDS = sorted(SCHED_REGISTRY.keys())


def build_scheduler(
    optimizer: Optimizer,
    sched_type: str,
    **kwargs: Any,
) -> "nn.Module":
    """Construct and return an LR scheduler wrapper for *sched_type*.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer whose learning rate will be scheduled.
    sched_type : str
        Scheduler identifier (e.g. ``'cosine'``, ``'step'``, ``'one_cycle'``).
    **kwargs
        Extra keyword arguments forwarded to the scheduler constructor.

    Returns
    -------
    nn.Module
        A scheduler wrapper instance (registered in ``SCHED_REGISTRY``).

    Raises
    ------
    ValueError
        If *sched_type* is not supported.
    """
    if not SCHED_REGISTRY.contains(sched_type):
        raise ValueError(
            f"Unknown scheduler {sched_type!r}. "
            f"Supported schedulers: {_SUPPORTED_SCHEDS}"
        )

    return SCHED_REGISTRY.build(sched_type, optimizer=optimizer, **kwargs)


__all__ = [
    "SCHED_REGISTRY",
    "register_scheduler",
    "build_scheduler",
    "CosineAnnealingLRWrapper",
    "StepLRWrapper",
    "OneCycleLRWrapper",
]
