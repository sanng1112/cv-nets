"""Tversky Loss for imbalanced segmentation.

Asymmetric similarity: ``TP / (TP + α·FP + β·FN)``.
When ``α = β = 0.5``, equivalent to Dice loss.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("tversky_loss", category="segmentation")
class TverskyLoss(BaseLoss):
    """Tversky loss for multi-class segmentation.

    Parameters
    ----------
    alpha : float
        Weight for false positives (default ``0.5``).
    beta : float
        Weight for false negatives (default ``0.5``).
    smooth : float
        Smoothing term to avoid division by zero (default ``1e-6``).
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """Compute Tversky loss.

        Parameters
        ----------
        prediction : Tensor
            Raw logits of shape ``(N, C, H, W)``.
        target : Tensor
            Ground-truth class indices of shape ``(N, H, W)``.

        Returns
        -------
        Tensor
            Reduced loss.
        """
        C = prediction.shape[1]
        p = F.softmax(prediction, dim=1)
        t = F.one_hot(target, C).permute(0, 3, 1, 2).float()

        # Flatten spatial dimensions
        p = p.contiguous().view(p.shape[0], C, -1)
        t = t.contiguous().view(t.shape[0], C, -1)

        tp = (p * t).sum(dim=2)
        fp = (p * (1.0 - t)).sum(dim=2)
        fn = ((1.0 - p) * t).sum(dim=2)

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        loss = 1.0 - tversky_index
        # Average over classes per sample
        loss = loss.mean(dim=1)

        return self._reduce(loss)

    def extra_repr(self) -> str:
        return (
            f"reduction={self.reduction}, "
            f"alpha={self.alpha}, "
            f"beta={self.beta}"
        )
