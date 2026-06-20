"""
Optimizer registry — factory dispatch for optimizers.

Usage
-----
>>> from cvnets.optim.registry import OPTIM_REGISTRY, register_optimizer
>>>
>>> @register_optimizer("sgd")
>>> class SGDWrapper(nn.Module):
>>>     ...
>>>
>>> optim = OPTIM_REGISTRY.build("sgd", model.parameters(), lr=0.01)
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from cvnets.core.registry import Registry

T = TypeVar("T", bound=Callable[..., Any])

OPTIM_REGISTRY = Registry("optimizer")
"""Global registry for optimizers."""


def register_optimizer(name: str) -> Callable[[T], T]:
    """Decorator: register a class under *name* in ``OPTIM_REGISTRY``.

    Parameters
    ----------
    name : str
        Identifier for the optimizer (e.g. ``'sgd'``, ``'adam'``).

    Returns
    -------
    Callable
        A decorator that registers the class and returns it unchanged.
    """
    return OPTIM_REGISTRY.register(name)
