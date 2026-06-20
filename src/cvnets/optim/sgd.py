"""
SGD optimizer wrapper.

Wraps ``torch.optim.SGD`` into an ``nn.Module`` so it can be registered and
built via the optimizer registry.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
from torch import Tensor, nn
from torch.optim import SGD

from cvnets.optim.registry import register_optimizer


@register_optimizer("sgd")
class SGDWrapper(nn.Module):
    """Thin ``nn.Module`` wrapper around ``torch.optim.SGD``.

    Parameters
    ----------
    params : iterable of parameters or dicts
        Model parameters (or parameter groups) to optimise.
    lr : float
        Learning rate.
    momentum : float
        Momentum factor (default 0.0).
    weight_decay : float
        Weight decay (L2 penalty, default 0.0).
    dampening : float
        Dampening for momentum (default 0.0).
    nesterov : bool
        Whether to use Nesterov momentum (default False).
    """

    def __init__(
        self,
        params: Iterable[Union[nn.parameter.Parameter, Dict[str, Any]]],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__()
        self._optimizer = SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )

    @property
    def optimizer(self) -> SGD:
        """Return the underlying ``torch.optim.SGD`` instance."""
        return self._optimizer

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        """Delegate to the underlying optimizer's param_groups."""
        return self._optimizer.param_groups

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step."""
        return self._optimizer.step(closure=closure)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Reset the gradients of all optimised parameters."""
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """Return the optimizer state dict."""
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the optimizer state dict."""
        self._optimizer.load_state_dict(state_dict)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._optimizer})"
