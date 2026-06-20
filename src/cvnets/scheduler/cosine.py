"""
Cosine annealing learning-rate scheduler wrapper.

Wraps ``torch.optim.lr_scheduler.CosineAnnealingLR`` into an ``nn.Module``
so it can be registered and built via the scheduler registry.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer

from cvnets.scheduler.registry import register_scheduler


@register_scheduler("cosine")
class CosineAnnealingLRWrapper(nn.Module):
    """Thin ``nn.Module`` wrapper around ``CosineAnnealingLR``.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    T_max : int
        Maximum number of iterations (epochs).
    eta_min : float
        Minimum learning rate (default 0.0).
    last_epoch : int
        The index of last epoch (default -1).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        super().__init__()
        self._scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )

    @property
    def scheduler(self) -> CosineAnnealingLR:
        """Return the underlying ``CosineAnnealingLR`` instance."""
        return self._scheduler

    def step(self) -> None:
        """Advance the scheduler by one step."""
        self._scheduler.step()

    def state_dict(self) -> Dict[str, Any]:
        """Return the scheduler state dict."""
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the scheduler state dict."""
        self._scheduler.load_state_dict(state_dict)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(T_max={self._scheduler.T_max})"
