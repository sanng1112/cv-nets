from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor


def reduce_loss(loss: Tensor, reduction: str = "mean", weight: Optional[Tensor] = None) -> Tensor:
    """Apply reduction to a per-element loss tensor.

    Parameters
    ----------
    loss : Tensor
        Per-element loss of shape ``(B,)`` or ``(B, ...)``.
    reduction : str
        One of ``'mean'``, ``'sum'``, ``'none'``.
    weight : Tensor or None
        Optional per-sample weight of shape ``(B,)``.

    Returns
    -------
    Tensor
        Reduced loss scalar or tensor.
    """
    if reduction == "none":
        return loss
    if weight is not None:
        loss = loss * weight
        if reduction == "mean":
            return loss.sum() / weight.sum().clamp(min=1e-8)
        return loss.sum()
    return loss.mean() if reduction == "mean" else loss.sum()
