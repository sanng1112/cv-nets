import argparse
import torch
import torch.nn.functional as F
from torch import Tensor

from loss_fn import LOSS_REGISTRY
from loss_fn.metric_learning.base_metric import BaseMetricCriteria

@LOSS_REGISTRY.register(name="contrastive", type="metric_learning")
class ContrastiveLoss(BaseMetricCriteria):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(opts, *args, **kwargs)
        self.margin = getattr(opts, "loss.metric_learning.contrastive.margin", 1.0)
        self.reduction = getattr(opts, "loss.metric_learning.contrastive.reduction", "mean")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if cls != ContrastiveLoss:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.metric-learning.contrastive.margin",
            type=float,
            default=1.0,
            help="Margin for Contrastive Loss.",
        )
        return parser

    def _compute_loss(self, output1: Tensor, output2: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        # target: 1 means similar, 0 means dissimilar
        """
        Chi tiết hàm: `_compute_loss`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - target) * torch.pow(euclidean_distance, 2) +
                                      (target) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

    def extra_repr(self) -> str:
        """
        Chi tiết hàm: `extra_repr`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"\n\t margin={self.margin}"
