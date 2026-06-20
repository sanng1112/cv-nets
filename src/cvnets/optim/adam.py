"""
Adam / AdamW optimizer wrappers.

Wraps ``torch.optim.Adam`` and ``torch.optim.AdamW`` into ``nn.Module``
subclasses so they can be registered and built via the optimizer registry.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
from torch import Tensor, nn
from torch.optim import Adam, AdamW

from cvnets.optim.registry import register_optimizer


# ===================================================================
# AdamWrapper
# ===================================================================


@register_optimizer("adam")
class AdamWrapper(nn.Module):
    """Thin ``nn.Module`` wrapper around ``torch.optim.Adam``.

    Parameters
    ----------
    params : iterable of parameters or dicts
        Model parameters (or parameter groups) to optimise.
    lr : float
        Learning rate.
    betas : tuple of (beta1, beta2)
        Coefficients for computing running averages of gradient and its
        square (default ``(0.9, 0.999)``).
    eps : float
        Term added to denominator to improve numerical stability.
    weight_decay : float
        Weight decay (L2 penalty, default 0.0).
    amsgrad : bool
        Whether to use the AMSGrad variant (default False).
    """

    def __init__(
        self,
        params: Iterable[Union[nn.parameter.Parameter, Dict[str, Any]]],
        lr: float,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        super().__init__()
        self._optimizer = Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

    @property
    def optimizer(self) -> Adam:
        """Return the underlying ``torch.optim.Adam`` instance."""
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


# ===================================================================
# AdamWWrapper
# ===================================================================


@register_optimizer("adamw")
class AdamWWrapper(nn.Module):
    """Thin ``nn.Module`` wrapper around ``torch.optim.AdamW``.

    Parameters
    ----------
    params : iterable of parameters or dicts
        Model parameters (or parameter groups) to optimise.
    lr : float
        Learning rate.
    betas : tuple of (beta1, beta2)
        Coefficients for computing running averages of gradient and its
        square (default ``(0.9, 0.999)``).
    eps : float
        Term added to denominator to improve numerical stability.
    weight_decay : float
        Weight decay (L2 penalty, default 0.0).
    amsgrad : bool
        Whether to use the AMSGrad variant (default False).
    """

    def __init__(
        self,
        params: Iterable[Union[nn.parameter.Parameter, Dict[str, Any]]],
        lr: float,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        super().__init__()
        self._optimizer = AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

    @property
    def optimizer(self) -> AdamW:
        """Return the underlying ``torch.optim.AdamW`` instance."""
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
