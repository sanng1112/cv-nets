import argparse
import torch.nn.functional as F
from torch import Tensor

from loss_fn import LOSS_REGISTRY
from loss_fn.metric_learning.base_metric import BaseMetricCriteria

@LOSS_REGISTRY.register(name="triplet", type="metric_learning")
class TripletMarginLoss(BaseMetricCriteria):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(opts, *args, **kwargs)
        self.margin = getattr(opts, "loss.metric_learning.triplet.margin", 1.0)
        self.p = getattr(opts, "loss.metric_learning.triplet.p", 2.0)
        self.reduction = getattr(opts, "loss.metric_learning.triplet.reduction", "mean")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if cls != TripletMarginLoss:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.metric-learning.triplet.margin",
            type=float,
            default=1.0,
            help="Margin for Triplet Margin Loss.",
        )
        group.add_argument(
            "--loss.metric-learning.triplet.p",
            type=float,
            default=2.0,
            help="The norm degree for pairwise distance.",
        )
        return parser

    def _compute_loss(self, anchor: Tensor, positive: Tensor, negative: Tensor, target: Tensor = None, *args, **kwargs) -> Tensor:
        """
        Chi tiết hàm: `_compute_loss`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return F.triplet_margin_loss(
            anchor, positive, negative, 
            margin=self.margin, p=self.p, reduction=self.reduction
        )

    def extra_repr(self) -> str:
        """
        Chi tiết hàm: `extra_repr`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"\n\t margin={self.margin}\n\t p={self.p}"
