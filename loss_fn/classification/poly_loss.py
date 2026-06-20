import argparse
import torch
from torch import Tensor
from torch.nn import functional as F

from loss_fn import LOSS_REGISTRY
from loss_fn.classification.base_classification import BaseClassificationCriteria

@LOSS_REGISTRY.register(name="poly_loss", type="classification")
class PolyLoss(BaseClassificationCriteria):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(opts, *args, **kwargs)
        self.epsilon = getattr(opts, "loss.classification.poly_loss.epsilon", 2.0)
        self.ignore_idx = getattr(opts, "loss.classification.poly_loss.ignore_index", -1)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if cls != PolyLoss:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.classification.poly-loss.epsilon",
            type=float,
            default=2.0,
            help="Epsilon parameter for PolyLoss. Defaults to 2.0.",
        )
        group.add_argument(
            "--loss.classification.poly-loss.ignore-index",
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
        poly1 = ce_loss + self.epsilon * (1 - pt)
        return poly1.mean()

    def extra_repr(self) -> str:
        """
        Chi tiết hàm: `extra_repr`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"\n\t epsilon={self.epsilon}\n\t ignore_idx={self.ignore_idx}"
