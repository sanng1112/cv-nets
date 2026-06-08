"""Lovász-Softmax loss for semantic segmentation.

A convex surrogate for the Jaccard index (IoU) based on the Lovász hinge.
Reference: Berman, Triki, Blaschko (CVPR 2018).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


def _lovasz_grad(gt_sorted: Tensor) -> Tensor:
    """Compute gradient of the Lovász hinge w.r.t. sorted errors.

    Parameters
    ----------
    gt_sorted : Tensor
        Sorted ground-truth (values in ``{0, 1}``) of shape ``(S,)``.

    Returns
    -------
    Tensor
        Gradient weights of shape ``(S,)``.
    """
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = intersection / union.clamp(min=1e-8)
    if gt_sorted.shape[0] > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


@register_loss_fn("lovasz_softmax", category="segmentation")
class LovaszSoftmax(BaseLoss):
    """Lovász-Softmax loss for multi-class segmentation.

    Surrogate for the Jaccard index (IoU) averaged over all classes.
    Uses the Lovász hinge with a subgradient that can be back-propagated
    through the sorting operation.

    Parameters
    ----------
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)

    def forward(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """Compute Lovász-Softmax loss.

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
        N, C, H, W = prediction.shape
        # Softmax probabilities
        prob = F.softmax(prediction, dim=1)  # (N, C, H, W)

        # Flatten spatial dimensions
        prob = prob.contiguous().view(N, C, -1)  # (N, C, S)
        target_flat = target.contiguous().view(N, -1)  # (N, S)

        losses = []
        for batch_idx in range(N):
            class_losses = []
            for c in range(C):
                # Binary labels for class c (1 where target == c, 0 elsewhere)
                labels = (target_flat[batch_idx] == c).float()
                # Errors:
                #   For foreground (label=1): 1 - prob (false negative)
                #   For background (label=0): prob (false positive)
                prob_c = prob[batch_idx, c]  # (S,)
                errors = torch.where(
                    labels == 1.0,
                    1.0 - prob_c,
                    prob_c,
                )
                # Sort errors descending
                errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
                gt_sorted = labels[perm]
                grad = _lovasz_grad(gt_sorted)
                loss_c = torch.dot(errors_sorted, grad)
                class_losses.append(loss_c)
            losses.append(torch.stack(class_losses).mean())

        loss = torch.stack(losses)  # (N,)

        return self._reduce(loss)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"
