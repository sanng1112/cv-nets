"""Triplet Loss with batch-hard / semi-hard / all mining strategies.

References
----------
- Schroff, Florian, Dmitry Kalenichenko, and James Philbin.
  "FaceNet: A Unified Embedding for Face Recognition and Clustering."
  CVPR 2015.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("triplet_loss", category="metric_learning")
class TripletLoss(BaseLoss):
    """Triplet loss with configurable mining strategy.

    Parameters
    ----------
    margin : float
        Margin for the triplet hinge loss (default ``1.0``).
    mining : str
        Mining strategy: ``'hard'``, ``'semi_hard'``, or ``'all'``
        (default ``'hard'``).
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(self, margin: float = 1.0, mining: str = "hard", reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        if margin < 0:
            raise ValueError(f"margin must be >= 0, got {margin}")
        if mining not in ("hard", "semi_hard", "all"):
            raise ValueError(f"mining must be 'hard', 'semi_hard', or 'all', got {mining!r}")
        self.margin = margin
        self.mining = mining

    def _pairwise_distances(self, emb: Tensor) -> Tensor:
        """Compute squared Euclidean distance matrix.

        Parameters
        ----------
        emb : Tensor
            Embeddings of shape ``(B, D)``.

        Returns
        -------
        Tensor
            Pairwise squared-distance matrix of shape ``(B, B)``.
        """
        # ‖a - b‖² = ‖a‖² + ‖b‖² - 2 a·b
        sq = (emb * emb).sum(dim=1)  # (B,)
        dot = emb @ emb.t()          # (B, B)
        return sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * dot

    def forward(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """Compute triplet loss.

        Parameters
        ----------
        prediction : Tensor
            Embeddings of shape ``(B, D)``.
        target : Tensor
            Class labels of shape ``(B,)``.

        Returns
        -------
        Tensor
            Reduced loss.
        """
        emb = F.normalize(prediction, dim=1)
        dist = self._pairwise_distances(emb)  # (B, B)
        B = emb.shape[0]
        device = emb.device

        # Masks
        eye = torch.eye(B, dtype=torch.bool, device=device)
        same = target.unsqueeze(0) == target.unsqueeze(1)  # (B, B)
        pos_mask = same & (~eye)   # valid positive pairs (i != j, same class)
        neg_mask = (~same) & (~eye)  # valid negative pairs (i != j, diff class)

        if self.mining == "hard":
            # Hard mining: hardest positive + hardest negative per anchor
            # Positive: max distance among same-class samples
            pos_mask_f = pos_mask.float()
            hardest_positive = (dist * pos_mask_f).max(dim=1).values  # (B,)
            # Negative: min distance among different-class samples
            # Use a large value to mask out invalid entries
            neg_invalid = (~neg_mask).float() * 1e12
            hardest_negative = (dist + neg_invalid).min(dim=1).values  # (B,)
            loss = torch.relu(hardest_positive - hardest_negative + self.margin)

        elif self.mining == "semi_hard":
            # Semi-hard: negatives farther than the positive but within margin
            # For each anchor i: positive j, negative k
            # semi-hard: dist[i,k] > dist[i,j] AND dist[i,k] < dist[i,j] + margin
            # We average over all semi-hard triplets
            pos_mask_f = pos_mask.float()

            # For each pair (i,j) where j is valid positive
            # Expand dims: dist[i,k] comparison
            pos_dist = dist.unsqueeze(2)  # (B, B, 1)
            neg_dist = dist.unsqueeze(1)  # (B, 1, B)

            # Triplet hinge: dist_ap - dist_an + margin
            triplet_hinge = pos_dist - neg_dist + self.margin  # (B, B, B)

            # Validity masks: i,j same class, i,k different class, j != i, k != i
            valid_triplet_mask = (
                pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
            ).float()  # (B, B, B)

            # Semi-hard constraint: dist_an > dist_ap (i.e., hinge > margin)
            # AND dist_an < dist_ap + margin (i.e., hinge < 2*margin)
            semi_hard_mask = (
                (triplet_hinge > self.margin) & (triplet_hinge < 2.0 * self.margin)
            ).float()

            combined_mask = valid_triplet_mask * semi_hard_mask
            triplet_loss = torch.relu(triplet_hinge) * combined_mask

            # Count valid triplets per anchor
            cnt = combined_mask.sum(dim=(1, 2)).clamp(min=1)
            loss = triplet_loss.sum(dim=(1, 2)) / cnt

        else:  # "all"
            # Average over all valid triplets
            pos_dist = dist.unsqueeze(2)   # (B, B, 1)
            neg_dist = dist.unsqueeze(1)   # (B, 1, B)
            triplet_hinge = pos_dist - neg_dist + self.margin  # (B, B, B)

            valid_triplet_mask = (
                pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
            ).float()

            triplet_loss = torch.relu(triplet_hinge) * valid_triplet_mask

            cnt = valid_triplet_mask.sum(dim=(1, 2)).clamp(min=1)
            loss = triplet_loss.sum(dim=(1, 2)) / cnt

        return self._reduce(loss)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}, margin={self.margin}, mining={self.mining}"
