"""Cross-Entropy Loss with label smoothing, ignore_index, and class_weight support."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("cross_entropy", category="classification")
class CrossEntropyLoss(BaseLoss):
    """Cross-entropy loss with optional label smoothing, ignore_index, and class weights.

    Parameters
    ----------
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    label_smoothing : float
        Label smoothing strength (default ``0.0``).
    ignore_index : int
        Target value to ignore (default ``-100``).
    class_weight : Tensor or None
        Per-class weights (default ``None``).
    """

    def __init__(
        self,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        class_weight: Optional[Tensor] = None,
    ) -> None:
        super().__init__(reduction=reduction)
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.class_weight = class_weight

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Compute cross-entropy loss.

        Parameters
        ----------
        prediction : Tensor
            Raw logits of shape ``(N, C, ...)``.
        target : Tensor
            Ground-truth class indices of shape ``(N, ...)``.

        Returns
        -------
        Tensor
            Reduced loss.
        """
        loss = F.cross_entropy(
            prediction,
            target,
            weight=self.class_weight,
            ignore_index=self.ignore_index,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        return self._reduce(loss)

    def extra_repr(self) -> str:
        return (
            f"reduction={self.reduction}, "
            f"label_smoothing={self.label_smoothing}, "
            f"ignore_index={self.ignore_index}"
        )
