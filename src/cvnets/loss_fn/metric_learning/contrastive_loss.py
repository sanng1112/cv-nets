"""Contrastive Loss for Siamese networks.

References
----------
- Chopra, Sumit, Raia Hadsell, and Yann LeCun.
  "Learning a Similarity Metric Discriminatively, with Application to
  Face Verification." CVPR 2005.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("contrastive_loss", category="metric_learning")
class ContrastiveLoss(BaseLoss):
    """Contrastive loss for Siamese / pair-based metric learning.

    Parameters
    ----------
    margin : float
        Margin separating negative pairs (default ``2.0``).
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(self, margin: float = 2.0, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.margin = margin

    def forward(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """Compute contrastive loss.

        Parameters
        ----------
        prediction : Tensor
            First embedding of shape ``(B, D)``.
        target : Tensor
            Second embedding of shape ``(B, D)``.
        label : Tensor
            Pair labels of shape ``(B,)``: ``1`` for positive, ``0`` for negative.
            Passed via ``**kwargs``.

        Returns
        -------
        Tensor
            Reduced loss.
        """
        label = kwargs.get("label", target)
        if not isinstance(label, Tensor):
            label = Tensor(label)
        label = label.float()

        # Euclidean distance per pair
        d = F.pairwise_distance(prediction, target)  # (B,)

        # Positive pairs: 0.5 * d^2
        pos = 0.5 * label * d ** 2
        # Negative pairs: 0.5 * max(0, margin - d)^2
        neg = 0.5 * (1.0 - label) * (torch.clamp(self.margin - d, min=0.0) ** 2)

        return self._reduce(pos + neg)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}, margin={self.margin}"
