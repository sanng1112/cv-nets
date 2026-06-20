import argparse
import torch
from torch import Tensor
from torch.nn import functional as F

from loss_fn import LOSS_REGISTRY
from loss_fn.classification.base_classification import BaseClassificationCriteria

@LOSS_REGISTRY.register(name="focal_loss", type="classification")
class FocalLoss(BaseClassificationCriteria):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(opts, *args, **kwargs)

        self.gamma = getattr(opts, "loss.classification.focal_loss.gamma", 2.0)
        self.alpha = getattr(opts, "loss.classification.focal_loss.alpha", 0.25)
        self.ignore_idx = getattr(opts, "loss.classification.focal_loss.ignore_index", -1)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if cls != FocalLoss:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.classification.focal-loss.gamma",
            type=float,
            default=2.0,
            help="Gamma parameter for Focal Loss. Defaults to 2.0.",
        )
        group.add_argument(
            "--loss.classification.focal-loss.alpha",
            type=float,
            default=0.25,
            help="Alpha parameter for Focal Loss. Defaults to 0.25.",
        )
        group.add_argument(
            "--loss.classification.focal-loss.ignore-index",
            type=int,
            default=-1,
            help="Target value that is ignored. Defaults to -1.",
        )
        return parser    

    def _compute_loss(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """
        Chi tiết hàm: `_compute_loss`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        ce_loss = F.cross_entropy(prediction, target, reduction='none', ignore_index=self.ignore_idx)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

    def extra_repr(self) -> str:
        """
        Chi tiết hàm: `extra_repr`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"\n\t gamma={self.gamma}\n\t alpha={self.alpha}\n\t ignore_idx={self.ignore_idx}"
