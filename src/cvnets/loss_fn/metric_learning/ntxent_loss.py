"""NT-Xent / InfoNCE Loss for contrastive learning (SimCLR).

References
----------
- Chen, Ting, et al. "A Simple Framework for Contrastive Learning of
  Visual Representations." ICML 2020.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("ntxent_loss", category="metric_learning")
class NTXentLoss(BaseLoss):
    """NT-Xent (Normalised Temperature-scaled Cross-Entropy) loss.

    Parameters
    ----------
    temperature : float
        Scale factor for logits (default ``0.5``). Must be > 0.
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(self, temperature: float = 0.5, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def forward(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """Compute NT-Xent loss.

        Parameters
        ----------
        prediction : Tensor
            Embeddings of shape ``(B, D)``.
        target : Tensor
            Positive pair indices of shape ``(B,)``. Samples with the
            same target value are considered positive pairs.

        Returns
        -------
        Tensor
            Reduced loss.
        """
        emb = F.normalize(prediction, dim=1)          # (B, D)
        B = emb.shape[0]
        device = emb.device

        # Cosine similarity matrix
        sim = emb @ emb.t() / self.temperature        # (B, B)

        # Remove diagonal (self-similarity)
        eye = torch.eye(B, dtype=torch.bool, device=device)
        sim_no_diag = sim[~eye].view(B, B - 1)         # (B, B-1)

        # Positive pair mask (same target, i != j)
        same = target.unsqueeze(0) == target.unsqueeze(1)  # (B, B)
        pos_mask = same & (~eye)
        pos_mask_no_diag = pos_mask[~eye].view(B, B - 1)   # (B, B-1)
        pos_count = pos_mask_no_diag.sum(dim=1).float().clamp(min=1)

        # Log-softmax over the non-diag similarities
        log_prob = F.log_softmax(sim_no_diag, dim=1)       # (B, B-1)

        # Gather positive log-probs and average per anchor
        pos_log_prob = (log_prob * pos_mask_no_diag.float()).sum(dim=1) / pos_count

        return self._reduce(-pos_log_prob)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}, temperature={self.temperature}"
