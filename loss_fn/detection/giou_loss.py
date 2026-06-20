import argparse
import torch
from torch import Tensor

from loss_fn import LOSS_REGISTRY
from loss_fn.detection.base_detection import BaseDetectionCriteria

@LOSS_REGISTRY.register(name="giou", type="detection")
class GIoULoss(BaseDetectionCriteria):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(opts, *args, **kwargs)
        self.reduction = getattr(opts, "loss.detection.giou.reduction", "mean")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if cls != GIoULoss:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.detection.giou.reduction",
            type=str,
            default="mean",
            choices=["mean", "sum", "none"],
            help="Reduction method for GIoU loss.",
        )
        return parser

    def _compute_loss(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        # prediction and target: [N, 4] in (x1, y1, x2, y2) format
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
        preds_area = (preds_right - preds_left) * (preds_bottom - preds_top)
        targets_area = (targets_right - targets_left) * (targets_bottom - targets_top)
        union = preds_area + targets_area - intersection

        iou = intersection / (union + self.eps)

        # Enclosing box
        enclose_left = torch.min(preds_left, targets_left)
        enclose_top = torch.min(preds_top, targets_top)
        enclose_right = torch.max(preds_right, targets_right)
        enclose_bottom = torch.max(preds_bottom, targets_bottom)

        enclose_w = torch.clamp(enclose_right - enclose_left, min=0)
        enclose_h = torch.clamp(enclose_bottom - enclose_top, min=0)
        enclose_area = enclose_w * enclose_h

        giou = iou - (enclose_area - union) / (enclose_area + self.eps)
        loss = 1 - giou

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
