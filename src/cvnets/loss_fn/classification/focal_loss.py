"""Focal Loss for imbalanced classification."""
from __future__ import annotations

from typing import List, Optional, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


@register_loss_fn("focal_loss", category="classification")
class FocalLoss(BaseLoss):
    """Focal loss for imbalanced multi-class classification.

    Parameters
    ----------
    gamma : float
        Focusing parameter (default ``2.0``). ``gamma=0`` is equivalent to
        standard cross-entropy.
    alpha : float or list of float or None
        Optional class weight(s). If a single float, it is used as the
        weight for the positive class in a binary setting; if a list, it
        provides per-class weights.
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    ignore_index : int
        Target value to ignore (default ``-100``).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, List[float]]] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__(reduction=reduction)
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Compute focal loss.

        Parameters
        ----------
        prediction : Tensor
            Raw logits of shape ``(N, C)``.
        target : Tensor
            Ground-truth class indices of shape ``(N,)``.

        Returns
        -------
        Tensor
            Reduced loss.
        """
        # Create mask for valid (non-ignored) targets
        mask = target != self.ignore_index
        # Clamp target so gather works even with ignore_index
        safe_target = target.clamp(min=0)

        log_prob = F.log_softmax(prediction, dim=-1)
        prob = log_prob.exp()

        # Gather per-sample log-prob of the target class
        log_pt = log_prob.gather(1, safe_target.unsqueeze(-1)).squeeze(-1)
        pt = prob.gather(1, safe_target.unsqueeze(-1)).squeeze(-1)

        # Focal modulation: (1 - pt) ^ gamma
        focal_weight = (1 - pt) ** self.gamma

        loss = -focal_weight * log_pt

        # Apply alpha balancing
        if self.alpha is not None:
            if isinstance(self.alpha, (list,)):
                alpha_t = torch.tensor(self.alpha, device=prediction.device).gather(0, safe_target)
            else:
                alpha_t = torch.full_like(target, self.alpha, dtype=torch.float)
            loss = loss * alpha_t

        # Zero out ignored positions
        loss = loss * mask.float()

        return self._reduce(loss)

    def extra_repr(self) -> str:
        return (
            f"reduction={self.reduction}, "
            f"gamma={self.gamma}, "
            f"alpha={self.alpha}, "
            f"ignore_index={self.ignore_index}"
        )
