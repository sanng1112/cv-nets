"""
StepLR scheduler wrapper.

Wraps ``torch.optim.lr_scheduler.StepLR`` into an ``nn.Module``
so it can be registered and built via the scheduler registry.
"""
from __future__ import annotations

from typing import Any, Dict

from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer

from cvnets.scheduler.registry import register_scheduler


@register_scheduler("step")
class StepLRWrapper(nn.Module):
    """Thin ``nn.Module`` wrapper around ``StepLR``.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    step_size : int
        Period of learning rate decay.
    gamma : float
        Multiplicative factor of learning rate decay (default 0.1).
    last_epoch : int
        The index of last epoch (default -1).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        super().__init__()
        self._scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch,
        )

    @property
    def scheduler(self) -> StepLR:
        """Return the underlying ``StepLR`` instance."""
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
        return f"{self.__class__.__name__}(step_size={self._scheduler.step_size})"
