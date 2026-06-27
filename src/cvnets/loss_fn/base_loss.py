from __future__ import annotations
import abc
import logging
from typing import Optional
import torch
from torch import Tensor, nn
from cvnets.loss_fn.reduction import reduce_loss

_logger = logging.getLogger(__name__)


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

    def _validate_inputs(self, prediction: Tensor, target: Tensor) -> None:
        """Validate prediction and target tensors before computing loss.

        Checks for NaN/Inf values and dtype consistency. Subclasses may
        override to add domain-specific checks.
        """
        if prediction.is_floating_point():
            if torch.isnan(prediction).any():
                raise ValueError("prediction tensor contains NaN values")
            if torch.isinf(prediction).any():
                raise ValueError("prediction tensor contains Inf values")

    def _reduce(self, loss: Tensor, weight: Optional[Tensor] = None) -> Tensor:
        """Apply the configured reduction to *loss*."""
        return reduce_loss(loss, reduction=self.reduction, weight=weight)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"
