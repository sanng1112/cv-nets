"""ArcFace Loss — additive angular margin loss for face recognition.

Reference: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
(Deng et al., 2019) https://arxiv.org/abs/1801.07698
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("arcface_loss", category="classification")
class ArcFaceLoss(BaseLoss):
    """ArcFace loss with learnable weight matrix and additive angular margin.

    Parameters
    ----------
    embed_dim : int
        Dimension of input embeddings.
    num_classes : int
        Number of classes.
    margin : float
        Angular margin in degrees (default ``0.5``).
    scale : float
        Feature scale / inverse temperature (default ``64.0``).
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_classes: int = 10,
        margin: float = 0.5,
        scale: float = 64.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        # Learnable weight matrix (rows = class centres)
        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings: Tensor, target: Tensor) -> Tensor:
        """Compute ArcFace loss.

        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings of shape ``(N, embed_dim)``.
        target : Tensor
            Ground-truth class indices of shape ``(N,)``.

        Returns
        -------
        Tensor
            Reduced loss.
        """
        # L2-normalise embeddings and weights
        embeddings = F.normalize(embeddings)
        weight = F.normalize(self.weight)

        # Cosine similarity: (N, C)
        cos_theta = torch.mm(embeddings, weight.t())  # (N, num_classes)

        # Clamp for numerical stability
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Compute sin(theta) from cos(theta)
        sin_theta = torch.sqrt(1.0 - cos_theta**2)

        # Add angular margin: cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Handle the case where cos(theta) > cos(pi - m) — use a different formula
        # to avoid negative values
        cond = cos_theta > self.th
        cos_theta_m = torch.where(
            cond, cos_theta_m, cos_theta - self.mm
        )

        # One-hot encode target
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()

        # Apply margin only to target positions
        logits = torch.where(
            one_hot.bool(),
            cos_theta_m,
            cos_theta,
        )

        # Scale logits
        logits = logits * self.scale

        # Cross-entropy loss
        loss = F.cross_entropy(logits, target, reduction="none")
        return self._reduce(loss)

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"num_classes={self.num_classes}, "
            f"margin={self.margin}, "
            f"scale={self.scale}, "
            f"reduction={self.reduction}"
        )
