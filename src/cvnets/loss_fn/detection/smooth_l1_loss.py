"""Smooth L1 Loss for object detection.

A variant of L1 loss that is quadratic for small errors and linear for large
errors, providing robustness to outliers.
"""

from __future__ import annotations

from torch.nn import functional as F

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("smooth_l1_loss", category="detection")
class SmoothL1Loss(BaseLoss):
    """Smooth L1 loss for bounding-box regression.

    Parameters
    ----------
    beta : float
        The threshold at which to switch between L2 and L1 (default ``1.0``).
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.beta = beta

    def forward(self, prediction, target, *args, **kwargs):
        """Compute Smooth L1 loss.

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
        return F.smooth_l1_loss(
            prediction, target, reduction=self.reduction, beta=self.beta
        )

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}, beta={self.beta}"
