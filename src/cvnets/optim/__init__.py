"""
Optimizer package for cv-nets.

Provides wrapper classes for PyTorch optimizers that can be registered
and built via the ``OPTIM_REGISTRY``. The main entry point is
``build_optimizer()``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Optional, Union

from torch import nn

from cvnets.optim.registry import OPTIM_REGISTRY, register_optimizer
from cvnets.optim.sgd import SGDWrapper
from cvnets.optim.adam import AdamWrapper, AdamWWrapper
from cvnets.utils.logger import info

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SUPPORTED_OPTIMS = sorted(OPTIM_REGISTRY.keys())


def build_optimizer(
    model_params: Union[Iterable[nn.parameter.Parameter], Iterator[nn.parameter.Parameter]],
    optim_type: str,
    verbose: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Construct and return an optimizer wrapper for *optim_type*.

    Parameters
    ----------
    model_params : iterable of parameters
        Model parameters to optimise.
    optim_type : str
        Optimizer identifier (e.g. ``'sgd'``, ``'adam'``, ``'adamw'``).
    verbose : bool
        If ``True``, log the optimizer configuration.
    **kwargs
        Extra keyword arguments forwarded to the optimizer constructor.

    Returns
    -------
    nn.Module
        An optimizer wrapper instance (registered in ``OPTIM_REGISTRY``).

    Raises
    ------
    ValueError
        If *optim_type* is not supported.
    """
    if not OPTIM_REGISTRY.contains(optim_type):
        raise ValueError(
            f"Unknown optimizer {optim_type!r}. "
            f"Supported optimizers: {_SUPPORTED_OPTIMS}"
        )

    if verbose:
        info(f"Building optimizer: {optim_type} with kwargs={kwargs}")

    return OPTIM_REGISTRY.build(optim_type, params=model_params, **kwargs)


__all__ = [
    "OPTIM_REGISTRY",
    "register_optimizer",
    "build_optimizer",
    "SGDWrapper",
    "AdamWrapper",
    "AdamWWrapper",
]
