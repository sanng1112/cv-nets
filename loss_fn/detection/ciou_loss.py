import argparse
import math
import torch
from torch import Tensor

from loss_fn import LOSS_REGISTRY
from loss_fn.detection.base_detection import BaseDetectionCriteria

@LOSS_REGISTRY.register(name="ciou", type="detection")
class CIoULoss(BaseDetectionCriteria):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(opts, *args, **kwargs)
        self.reduction = getattr(opts, "loss.detection.ciou.reduction", "mean")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if cls != CIoULoss:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.detection.ciou.reduction",
            type=str,
            default="mean",
            choices=["mean", "sum", "none"],
            help="Reduction method for CIoU loss.",
        )
        return parser

    def _compute_loss(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """
        Chi tiết hàm: `_compute_loss`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        preds_left = prediction[:, 0]
        preds_top = prediction[:, 1]
        preds_right = prediction[:, 2]
        preds_bottom = prediction[:, 3]

        targets_left = target[:, 0]
        targets_top = target[:, 1]
        targets_right = target[:, 2]
        targets_bottom = target[:, 3]

        # Intersect
        intersect_left = torch.max(preds_left, targets_left)
        intersect_top = torch.max(preds_top, targets_top)
        intersect_right = torch.min(preds_right, targets_right)
        intersect_bottom = torch.min(preds_bottom, targets_bottom)

        intersect_w = torch.clamp(intersect_right - intersect_left, min=0)
        intersect_h = torch.clamp(intersect_bottom - intersect_top, min=0)
        intersection = intersect_w * intersect_h

        # Areas
        preds_w = preds_right - preds_left
        preds_h = preds_bottom - preds_top
        preds_area = preds_w * preds_h

        targets_w = targets_right - targets_left
        targets_h = targets_bottom - targets_top
        targets_area = targets_w * targets_h

        union = preds_area + targets_area - intersection
        iou = intersection / (union + self.eps)

        # Enclosing box
        enclose_left = torch.min(preds_left, targets_left)
        enclose_top = torch.min(preds_top, targets_top)
        enclose_right = torch.max(preds_right, targets_right)
        enclose_bottom = torch.max(preds_bottom, targets_bottom)

        enclose_w = torch.clamp(enclose_right - enclose_left, min=0)
        enclose_h = torch.clamp(enclose_bottom - enclose_top, min=0)
        c2 = enclose_w**2 + enclose_h**2 + self.eps

        # Center distance
        preds_cx = (preds_left + preds_right) / 2
        preds_cy = (preds_top + preds_bottom) / 2
        targets_cx = (targets_left + targets_right) / 2
        targets_cy = (targets_top + targets_bottom) / 2

        d2 = (preds_cx - targets_cx)**2 + (preds_cy - targets_cy)**2

        # CIoU specific: aspect ratio penalty (v)
        v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(targets_w / (targets_h + self.eps)) - torch.atan(preds_w / (preds_h + self.eps)), 2)
        
        with torch.no_grad():
            alpha = v / ((1 - iou) + v + self.eps)

        ciou = iou - (d2 / c2 + alpha * v)
        loss = 1 - ciou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        """
        Chi tiết hàm: `extra_repr`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"\n\t reduction={self.reduction}"
