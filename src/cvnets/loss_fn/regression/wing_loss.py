"""Wing Loss for landmark regression.

Uses a logarithmic non-linearity for small errors and a linear function for
large errors, designed for robust facial landmark localisation.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("wing_loss", category="regression")
class WingLoss(BaseLoss):
    """Wing loss for (facial) landmark regression.

    ``loss = w * ln(1 + |x|/ε)`` for ``|x| < w``, and
    ``loss = |x| - C`` for ``|x| >= w``, where
    ``C = w - w * ln(1 + w/ε)``.

    Parameters
    ----------
    width : float
        The threshold at which to switch from the non-linear to the linear
        branch (default ``10.0``).
    epsilon : float
        Curvature parameter in the non-linear region (default ``2.0``).
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        width: float = 10.0,
        epsilon: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.width = width
        self.epsilon = epsilon

    def forward(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """Compute Wing loss.

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
        diff = (prediction - target).abs()
        # Constant ensuring continuity at |x| = width
        width = torch.tensor(self.width, device=diff.device, dtype=diff.dtype)
        eps = torch.tensor(self.epsilon, device=diff.device, dtype=diff.dtype)
        C = width - width * torch.log(1.0 + width / eps)

        mask = diff < width
        loss = torch.where(
            mask,
            width * torch.log(1.0 + diff / eps),
            diff - C,
        )
        return self._reduce(loss)

    def extra_repr(self) -> str:
        return (
            f"reduction={self.reduction}, "
            f"width={self.width}, epsilon={self.epsilon}"
        )
