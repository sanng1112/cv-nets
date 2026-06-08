"""Quantile Loss for regression (pinball loss).

Used for quantile regression and prediction intervals. Asymmetric loss that
penalises under- and over-predictions differently based on the target quantile.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("quantile_loss", category="regression")
class QuantileLoss(BaseLoss):
    """Quantile (pinball) loss for quantile regression.

    Parameters
    ----------
    quantile : float
        Target quantile in ``(0, 1)`` (default ``0.5``).
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        quantile: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile!r}")
        self.quantile = quantile

    def forward(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """Compute quantile (pinball) loss.

        ``loss = max(q * (y - p), (q - 1) * (y - p))``

        Parameters
        ----------
        prediction : Tensor
            Predicted values of any shape.
        target : Tensor
            Ground-truth values of the same shape.

        Returns
        -------
        Tensor
            Reduced loss.
        """
        error = target - prediction
        loss = torch.max(
            self.quantile * error,
            (self.quantile - 1.0) * error,
        )
        return self._reduce(loss)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}, quantile={self.quantile}"
