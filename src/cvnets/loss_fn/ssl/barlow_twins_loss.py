"""Barlow Twins Loss for self-supervised learning.

References
----------
- Zbontar, Jure, et al. "Barlow Twins: Self-Supervised Learning via
  Redundancy Reduction." ICML 2021.
"""
from __future__ import annotations

import torch
from torch import Tensor

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("barlow_twins_loss", category="ssl")
class BarlowTwinsLoss(BaseLoss):
    """Barlow Twins loss that pushes the cross-correlation matrix towards
    the identity.

    The loss has two components:
    - **Invariance term**: pulls diagonal elements toward 1.
    - **Redundancy-reduction term**: pushes off-diagonal elements toward 0.

    Parameters
    ----------
    lambd : float
        Weight for the off-diagonal (redundancy reduction) term
        (default ``0.005``).
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(self, lambd: float = 0.005, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.lambd = lambd

    def forward(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """Compute Barlow Twins loss.

        Parameters
        ----------
        prediction : Tensor
            Embeddings from first branch, shape ``(B, D)``.
        target : Tensor
            Embeddings from second branch, shape ``(B, D)``.

        Returns
        -------
        Tensor
            Reduced loss.
        """
        B, D = prediction.shape

        # Normalise along batch dimension (z-score)
        z1 = (prediction - prediction.mean(dim=0)) / prediction.std(dim=0, unbiased=False).clamp(min=1e-8)
        z2 = (target - target.mean(dim=0)) / target.std(dim=0, unbiased=False).clamp(min=1e-8)

        # Cross-correlation matrix (B, D) -> (D, D)
        c = (z1.T @ z2) / B

        # Diagonal → 1, off-diagonal → 0
        diag = torch.eye(D, device=c.device)
        on_diag = (c * diag).sum(dim=1)          # extract diagonal elements
        off_diag = (c * (1.0 - diag)).sum(dim=1)  # extract off-diagonal elements

        loss = (1.0 - on_diag).pow(2).sum() + self.lambd * off_diag.pow(2).sum()
        return self._reduce(loss.unsqueeze(0))

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}, lambd={self.lambd}"
