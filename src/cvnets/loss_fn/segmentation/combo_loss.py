"""Combo Loss for segmentation.

Weighted combination of Cross-Entropy and Dice losses:
``loss = α · CE + (1 − α) · Dice``
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("combo_loss", category="segmentation")
class ComboLoss(BaseLoss):
    """Combo loss: weighted Cross-Entropy + Dice.

    Parameters
    ----------
    alpha : float
        Weight for the cross-entropy term (default ``0.5``).
        The Dice term gets weight ``1 - alpha``.
    smooth : float
        Smoothing term for Dice (default ``1e-6``).
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        smooth: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """Compute Combo loss.

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
        # --- Cross-Entropy term (per-pixel, then average per sample) ---
        ce_per_pixel = F.cross_entropy(prediction, target, reduction="none")
        # ce_per_pixel: (N, H, W) -> average over spatial dims -> (N,)
        ce_per_sample = ce_per_pixel.view(ce_per_pixel.shape[0], -1).mean(dim=1)

        # --- Dice term (per sample) ---
        N, C, H, W = prediction.shape
        p = F.softmax(prediction, dim=1)
        t = F.one_hot(target, C).permute(0, 3, 1, 2).float()

        p_flat = p.contiguous().view(N, C, -1)
        t_flat = t.contiguous().view(N, C, -1)

        intersection = (p_flat * t_flat).sum(dim=2)
        dice_per_class = (2.0 * intersection + self.smooth) / (
            p_flat.sum(dim=2) + t_flat.sum(dim=2) + self.smooth
        )
        dice_per_sample = (1.0 - dice_per_class).mean(dim=1)  # (N,)

        # --- Combine ---
        loss = self.alpha * ce_per_sample + (1.0 - self.alpha) * dice_per_sample

        return self._reduce(loss)

    def extra_repr(self) -> str:
        return (
            f"reduction={self.reduction}, "
            f"alpha={self.alpha}, "
            f"smooth={self.smooth}"
        )
