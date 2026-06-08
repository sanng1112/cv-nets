"""Negative-Free Loss for BYOL and SimSiam self-supervised learning.

References
----------
- Grill, Jean-Bastien, et al. "Bootstrap Your Own Latent: A New Approach
  to Self-Supervised Learning." NeurIPS 2020.
- Chen, Xinlei, and Kaiming He. "Exploring Simple Siamese Representation
  Learning." CVPR 2021.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("negative_free_loss", category="ssl")
class NegativeFreeLoss(BaseLoss):
    """Negative-free loss for BYOL or SimSiam.

    Computes ``2 - 2 * cos(prediction, target)`` per sample.
    In BYOL mode the gradient is stopped on the *target* branch.

    Parameters
    ----------
    mode : str
        ``'byol'`` (detach target) or ``'simsiam'`` (gradient everywhere).
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(self, mode: str = "byol", reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        if mode not in ("byol", "simsiam"):
            raise ValueError(f"mode must be 'byol' or 'simsiam', got {mode!r}")
        self.mode = mode

    def forward(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """Compute negative-free loss.

        Parameters
        ----------
        prediction : Tensor
            Online/prediction embeddings of shape ``(B, D)``.
        target : Tensor
            Target embeddings of shape ``(B, D)``.

        Returns
        -------
        Tensor
            Reduced loss.
        """
        p = F.normalize(prediction, dim=1)
        with torch.no_grad() if self.mode == "byol" else torch.enable_grad():
            z = F.normalize(target, dim=1)
        # Cosine distance: 2 - 2 * cos(p, z)
        loss = 2.0 - 2.0 * (p * z).sum(dim=1)
        return self._reduce(loss)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}, mode={self.mode}"
