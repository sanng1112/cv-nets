"""Dice Loss for segmentation.

Supports multi-class (softmax) and binary (sigmoid) modes.
Dice coefficient: ``2 * |A ∩ B| / (|A| + |B|)``.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("dice_loss", category="segmentation")
class DiceLoss(BaseLoss):
    """Dice loss for binary and multi-class segmentation.

    Parameters
    ----------
    smooth : float
        Smoothing term to avoid division by zero (default ``1e-6``).
    binary : bool
        If ``True``, use sigmoid activation for binary segmentation.
        If ``False`` (default), use softmax over channels for multi-class.
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        smooth: float = 1e-6,
        binary: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.smooth = smooth
        self.binary = binary

    def forward(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """Compute Dice loss.

        Parameters
        ----------
        prediction : Tensor
            Logits of shape ``(N, C, H, W)`` for multi-class or
            ``(N, 1, H, W)`` for binary.
        target : Tensor
            Ground-truth of shape ``(N, H, W)`` with class indices for
            multi-class, or ``(N, 1, H, W)`` with float values in ``[0, 1]``
            for binary.

        Returns
        -------
        Tensor
            Reduced loss.
        """
        if self.binary:
            p = torch.sigmoid(prediction)
            p = p.contiguous().view(p.shape[0], -1)
            t = target.contiguous().view(target.shape[0], -1)
            intersection = (p * t).sum(dim=1)
            dice = (2.0 * intersection + self.smooth) / (
                p.sum(dim=1) + t.sum(dim=1) + self.smooth
            )
            loss = 1.0 - dice
        else:
            C = prediction.shape[1]
            p = F.softmax(prediction, dim=1)
            t = F.one_hot(target, C).permute(0, 3, 1, 2).float()
            p = p.contiguous().view(p.shape[0], C, -1)
            t = t.contiguous().view(t.shape[0], C, -1)
            intersection = (p * t).sum(dim=2)
            dice = (2.0 * intersection + self.smooth) / (
                p.sum(dim=2) + t.sum(dim=2) + self.smooth
            )
            loss = 1.0 - dice
            # Average over classes per sample
            loss = loss.mean(dim=1)

        return self._reduce(loss)

    def extra_repr(self) -> str:
        return (
            f"reduction={self.reduction}, "
            f"smooth={self.smooth}, "
            f"binary={self.binary}"
        )
