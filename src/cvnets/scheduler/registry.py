"""
Scheduler registry — factory dispatch for learning-rate schedulers.

Usage
-----
>>> from cvnets.scheduler.registry import SCHED_REGISTRY, register_scheduler
>>>
>>> @register_scheduler("cosine")
>>> class CosineAnnealingLRWrapper(nn.Module):
>>>     ...
>>>
>>> sched = SCHED_REGISTRY.build("cosine", optimizer, T_max=10)
"""
from __future__ import annotations

from typing import Any, Callable, TypeVar

from cvnets.core.registry import Registry

T = TypeVar("T", bound=Callable[..., Any])

SCHED_REGISTRY = Registry("scheduler")
"""Global registry for learning-rate schedulers."""


def register_scheduler(name: str) -> Callable[[T], T]:
    """Decorator: register a class under *name* in ``SCHED_REGISTRY``.

    Parameters
    ----------
    name : str
        Identifier for the scheduler (e.g. ``'cosine'``, ``'step'``).

    Returns
    -------
    Callable
        A decorator that registers the class and returns it unchanged.
    """
    return SCHED_REGISTRY.register(name)
