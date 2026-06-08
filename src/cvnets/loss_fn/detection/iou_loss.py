"""IoU / GIoU / DIoU / CIoU loss for object detection.

Computes intersection-over-union metrics between predicted and target
bounding boxes in ``(x1, y1, x2, y2)`` or ``(cx, cy, w, h)`` format.
Loss = ``1 - iou_value``, with optional GIoU/DIoU/CIoU corrections.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from cvnets.loss_fn import register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


def _box_iou(b1: Tensor, b2: Tensor) -> Tensor:
    """Compute pairwise IoU between two batches of axis-aligned boxes.

    Parameters
    ----------
    b1, b2 : Tensor
        Boxes in ``(x1, y1, x2, y2)`` format with shape ``(B, 4)``.

    Returns
    -------
    Tensor
        IoU values of shape ``(B,)``.
    """
    # Intersection coordinates
    ix1 = torch.max(b1[:, 0], b2[:, 0])
    iy1 = torch.max(b1[:, 1], b2[:, 1])
    ix2 = torch.min(b1[:, 2], b2[:, 2])
    iy2 = torch.min(b1[:, 3], b2[:, 3])

    # Intersection area
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih

    # Area of each box
    a1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    a2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)

    # Union
    union = a1 + a2 - inter

    return inter / union.clamp(min=1e-8)


def _inter_area(pred: Tensor, tgt: Tensor) -> Tensor:
    """Compute intersection area between pred and tgt boxes (helper for GIoU)."""
    ix1 = torch.max(pred[:, 0], tgt[:, 0])
    iy1 = torch.max(pred[:, 1], tgt[:, 1])
    ix2 = torch.min(pred[:, 2], tgt[:, 2])
    iy2 = torch.min(pred[:, 3], tgt[:, 3])
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    return iw * ih

@register_loss_fn("iou_loss", category="detection")
class IoULoss(BaseLoss):
    """IoU-based loss with GIoU / DIoU / CIoU variants.

    Parameters
    ----------
    mode : str
        One of ``'iou'``, ``'giou'``, ``'diou'``, ``'ciou'``.
    box_format : str
        ``'xyxy'`` (default) or ``'cxcywh'``.
    reduction : str
        Reduction method: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        mode: str = "iou",
        box_format: str = "xyxy",
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        if mode not in ("iou", "giou", "diou", "ciou"):
            raise ValueError(f"Unknown mode {mode!r}; expected iou/giou/diou/ciou")
        self.mode = mode
        self.box_format = box_format

    def forward(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """Compute IoU loss.

        Parameters
        ----------
        prediction : Tensor
            Predicted boxes of shape ``(B, 4)``.
        target : Tensor
            Ground-truth boxes of shape ``(B, 4)``.

        Returns
        -------
        Tensor
            Reduced loss.
        """
        # Convert cxcywh -> xyxy if needed
        if self.box_format == "cxcywh":
            pred = torch.stack(
                [
                    prediction[:, 0] - prediction[:, 2] / 2,
                    prediction[:, 1] - prediction[:, 3] / 2,
                    prediction[:, 0] + prediction[:, 2] / 2,
                    prediction[:, 1] + prediction[:, 3] / 2,
                ],
                dim=1,
            )
            tgt = torch.stack(
                [
                    target[:, 0] - target[:, 2] / 2,
                    target[:, 1] - target[:, 3] / 2,
                    target[:, 0] + target[:, 2] / 2,
                    target[:, 1] + target[:, 3] / 2,
                ],
                dim=1,
            )
        else:
            pred = prediction
            tgt = target

        # Base IoU
        iou = _box_iou(pred, tgt)

        if self.mode == "iou":
            loss = 1.0 - iou

        elif self.mode == "giou":
            # Convex hull (smallest enclosing box)
            cx1 = torch.min(pred[:, 0], tgt[:, 0])
            cy1 = torch.min(pred[:, 1], tgt[:, 1])
            cx2 = torch.max(pred[:, 2], tgt[:, 2])
            cy2 = torch.max(pred[:, 3], tgt[:, 3])
            c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)

            # Areas of pred and target boxes
            a1 = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (
                pred[:, 3] - pred[:, 1]
            ).clamp(min=0)
            a2 = (tgt[:, 2] - tgt[:, 0]).clamp(min=0) * (
                tgt[:, 3] - tgt[:, 1]
            ).clamp(min=0)
            inter = _inter_area(pred, tgt)
            u = a1 + a2 - inter

            giou = iou - (c_area - u).clamp(min=0) / c_area.clamp(min=1e-8)
            loss = 1.0 - giou

        else:  # diou or ciou
            # Unbind coordinates
            px1, py1, px2, py2 = pred.unbind(dim=-1)
            tx1, ty1, tx2, ty2 = tgt.unbind(dim=-1)

            # Center points
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2
            tcx = (tx1 + tx2) / 2
            tcy = (ty1 + ty2) / 2

            # Diagonal length of smallest enclosing box
            cx1 = torch.min(px1, tx1)
            cy1 = torch.min(py1, ty1)
            cx2 = torch.max(px2, tx2)
            cy2 = torch.max(py2, ty2)
            c_diag = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
            d_center = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

            diou_term = d_center / c_diag.clamp(min=1e-8)

            if self.mode == "diou":
                loss = 1.0 - iou + diou_term
            else:  # ciou
                pw = (px2 - px1).clamp(min=1e-8)
                ph = (py2 - py1).clamp(min=1e-8)
                tw = (tx2 - tx1).clamp(min=1e-8)
                th = (ty2 - ty1).clamp(min=1e-8)

                v = (4 / (math.pi**2)) * (
                    torch.atan(tw / th) - torch.atan(pw / ph)
                ) ** 2
                alpha = v / ((1.0 - iou) + v).clamp(min=1e-8)
                loss = 1.0 - iou + diou_term + alpha * v

        return self._reduce(loss)

    def extra_repr(self) -> str:
        return (
            f"reduction={self.reduction}, mode={self.mode}, "
            f"box_format={self.box_format}"
        )

