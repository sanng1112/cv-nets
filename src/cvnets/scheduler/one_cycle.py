"""
OneCycleLR scheduler wrapper.

Wraps ``torch.optim.lr_scheduler.OneCycleLR`` into an ``nn.Module``
so it can be registered and built via the scheduler registry.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.optimizer import Optimizer

from cvnets.scheduler.registry import register_scheduler


@register_scheduler("one_cycle")
class OneCycleLRWrapper(nn.Module):
    """Thin ``nn.Module`` wrapper around ``OneCycleLR``.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    max_lr : float
        Upper learning rate boundaries in the cycle.
    total_steps : int
        The total number of steps in the cycle. Note that if a value is not
        provided here, then it must be inferred by providing ``steps_per_epoch``
        and ``epochs``.
    epochs : int, optional
        The number of epochs to train for (used if *total_steps* is ``None``).
    steps_per_epoch : int, optional
        The number of steps per epoch (used if *total_steps* is ``None``).
    pct_start : float
        The percentage of the cycle (in number of steps) spent increasing the
        learning rate (default 0.3).
    anneal_strategy : str
        ``'cos'`` for cosine annealing, ``'linear'`` for linear annealing
        (default ``'cos'``).
    div_factor : float
        Initial LR = ``max_lr / div_factor`` (default 25.0).
    final_div_factor : float
        Final LR = ``max_lr / final_div_factor`` (default 1e4).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
    ) -> None:
        super().__init__()
        self._scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )

    @property
    def scheduler(self) -> OneCycleLR:
        """Return the underlying ``OneCycleLR`` instance."""
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
        return f"{self.__class__.__name__}(max_lr={self._scheduler.max_lr})"
