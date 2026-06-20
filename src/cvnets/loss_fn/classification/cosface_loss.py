"""CosFace Loss — additive cosine margin loss for face recognition.

Reference: "CosFace: Large Margin Cosine Loss for Deep Face Recognition"
(Wang et al., 2018) https://arxiv.org/abs/1801.09414
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("cosface_loss", category="classification")
class CosFaceLoss(BaseLoss):
    """CosFace loss with learnable weight matrix and additive cosine margin.

    Parameters
    ----------
    embed_dim : int
        Dimension of input embeddings.
    num_classes : int
        Number of classes.
    margin : float
        Cosine margin (default ``0.35``). Subtracted from the cosine
        similarity of the target class.
    scale : float
        Feature scale / inverse temperature (default ``64.0``).
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_classes: int = 10,
        margin: float = 0.35,
        scale: float = 64.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Learnable weight matrix (rows = class centres)
        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings: Tensor, target: Tensor) -> Tensor:
        """Compute CosFace loss.

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

        # Apply cosine margin: subtract margin from target class similarity
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        logits = cos_theta - one_hot * self.margin

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
