"""Asymmetric Loss for multi-label classification.

Reference: "Asymmetric Loss For Multi-Label Classification" (Ridnik et al., 2021)
https://arxiv.org/abs/2009.14119
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("asymmetric_loss", category="classification")
class AsymmetricLoss(BaseLoss):
    """Asymmetric loss (ASL) for multi-label classification.

    Applies different focusing exponents to positive and negative samples
    to handle class imbalance.

    Parameters
    ----------
    gamma_pos : float
        Focusing parameter for positive samples (default ``0.0``).
    gamma_neg : float
        Focusing parameter for negative samples (default ``4.0``).
    clip : float
        Probability clipping value to prevent over-suppression of
        easy negatives (default ``0.05``).
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Compute asymmetric loss.

        Parameters
        ----------
        prediction : Tensor
            Raw logits of shape ``(N, C)`` — one score per class.
        target : Tensor
            Binary targets of shape ``(N, C)`` with values in ``{0, 1}``.

        Returns
        -------
        Tensor
            Reduced loss.
        """
        # Compute probabilities and clip to avoid zero gradients
        prob = torch.sigmoid(prediction)
        if self.clip > 0:
            prob = prob.clamp(min=self.clip, max=1 - self.clip)

        # Separate positive and negative probabilities
        # pt = p if y=1, else 1-p if y=0
        pt = target * prob + (1 - target) * (1 - prob)

        # Asymmetric focusing: different gamma for pos vs neg
        gamma = target * self.gamma_pos + (1 - target) * self.gamma_neg
        focal_weight = (1 - pt) ** gamma

        # Binary cross-entropy with predicted probability
        loss = -focal_weight * torch.log(pt)

        # Average over classes for each sample
        loss = loss.mean(dim=-1)

        return self._reduce(loss)

    def extra_repr(self) -> str:
        return (
            f"reduction={self.reduction}, "
            f"gamma_pos={self.gamma_pos}, "
            f"gamma_neg={self.gamma_neg}, "
            f"clip={self.clip}"
        )
