"""Huber Loss for regression.

Combines quadratic behaviour for small errors with linear behaviour for large
errors, providing robustness to outliers.
"""

from __future__ import annotations

from torch.nn import functional as F

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("huber_loss", category="regression")
class HuberLoss(BaseLoss):
    """Huber loss for robust regression.

    Parameters
    ----------
    delta : float
        The threshold at which to switch between L2 and L1 (default ``1.0``).
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        delta: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.delta = delta

    def forward(self, prediction, target, *args, **kwargs):
        """Compute Huber loss.

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
        return F.huber_loss(
            prediction, target, reduction=self.reduction, delta=self.delta
        )

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}, delta={self.delta}"
