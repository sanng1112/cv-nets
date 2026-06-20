"""Circle Loss for unified metric learning.

References
----------
- Sun, Yifan, et al. "Circle Loss: A Unified Perspective of Pair
  Similarity Optimization." CVPR 2020.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("circle_loss", category="metric_learning")
class CircleLoss(BaseLoss):
    """Circle loss with adaptive weighting.

    Parameters
    ----------
    margin : float
        Margin parameter (default ``0.25``).
    gamma : int
        Scale factor (default ``80``).
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(self, margin: float = 0.25, gamma: int = 80, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.margin = margin
        self.gamma = gamma

    def forward(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """Compute circle loss.

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
        emb = F.normalize(prediction, dim=1)          # (B, D)
        sim = emb @ emb.t()                           # (B, B)
        B = emb.shape[0]
        device = emb.device

        # Masks
        eye = torch.eye(B, dtype=torch.bool, device=device)
        same = target.unsqueeze(0) == target.unsqueeze(1)  # (B, B)
        pos_mask = same & (~eye)   # positive pairs (i != j, same class)
        neg_mask = (~same) & (~eye)  # negative pairs (i != j, diff class)

        # Gather positive and negative similarities per anchor
        # For each anchor i, we have a set of positive and negative similarities
        pos_sim = sim * pos_mask.float()   # (B, B)
        neg_sim = sim * neg_mask.float()   # (B, B)

        # Prepare for batched computation: accumulate per anchor
        # Constants
        dp = 1.0 - self.margin   # optimal positive similarity
        dn = self.margin         # optimal negative similarity

        # Adaptive weights (stop-gradient on the similarity term)
        # α_p^j = ReLU(1 + margin - s_p^j)  i.e., ReLU(dp_star - s_p^j)
        # where dp_star = 1 + margin
        with torch.no_grad():
            ap = torch.relu(-pos_sim + 1.0 + self.margin)   # (B, B)
            an = torch.relu(neg_sim + self.margin)           # (B, B)

        # Compute per-anchor loss
        # L = log(1 + Σ exp(γ * α_n * (s_n - dn))) + log(1 + Σ exp(-γ * α_p * (s_p - dp)))
        eps = 1e-12

        # Positive term: -γ * α_p * (s_p - dp)
        pos_logits = -self.gamma * ap * (pos_sim - dp)   # (B, B)
        # Negative term: γ * α_n * (s_n - dn)
        neg_logits = self.gamma * an * (neg_sim - dn)    # (B, B)

        # Mask out invalid entries with -inf for positives and -inf for negatives
        # so they don't contribute to the log-sum-exp
        pos_logits = pos_logits * pos_mask.float() - 1e12 * (~pos_mask).float()
        neg_logits = neg_logits * neg_mask.float() - 1e12 * (~neg_mask).float()

        # Compute loss per anchor
        # Using log(1 + exp(x)) approximation for numerical stability
        pos_lse = torch.logsumexp(pos_logits, dim=1).clamp(max=80.0)  # (B,)
        neg_lse = torch.logsumexp(neg_logits, dim=1).clamp(max=80.0)  # (B,)

        loss = torch.relu(pos_lse + neg_lse)

        # Handle edge cases where no positive pairs exist
        has_pos = pos_mask.any(dim=1)
        has_neg = neg_mask.any(dim=1)
        valid = has_pos & has_neg
        loss = loss * valid.float()

        return self._reduce(loss)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}, margin={self.margin}, gamma={self.gamma}"
