from __future__ import annotations
import abc
from typing import Optional
from torch import Tensor, nn
from cvnets.loss_fn.reduction import reduce_loss


class BaseLoss(nn.Module, abc.ABC):
    """Abstract base for all cv-nets loss functions.

    Parameters
    ----------
    reduction : str
        How to reduce per-element losses: ``'mean'``, ``'sum'``, ``'none'``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean'/'sum'/'none', got {reduction!r}")
        self.reduction = reduction

    @abc.abstractmethod
    def forward(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """Compute the per-element loss, then apply reduction."""

    def _reduce(self, loss: Tensor, weight: Optional[Tensor] = None) -> Tensor:
        """Apply the configured reduction to *loss*."""
        return reduce_loss(loss, reduction=self.reduction, weight=weight)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"
