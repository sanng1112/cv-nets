import argparse
import torch
from torch import Tensor

from loss_fn import LOSS_REGISTRY
from loss_fn.generative.base_generative import BaseGenerativeCriteria

@LOSS_REGISTRY.register(name="tv", type="generative")
class TotalVariationLoss(BaseGenerativeCriteria):
    """
    Total Variation (TV) Loss.
    Encourages spatial smoothness in generated images.
    Target is usually None for TV loss.
    """
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(opts, *args, **kwargs)
        self.weight = getattr(opts, "loss.generative.tv.weight", 1.0)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if cls != TotalVariationLoss:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.generative.tv.weight",
            type=float,
            default=1.0,
            help="Weight for TV Loss.",
        )
        return parser

    def _compute_loss(self, prediction: Tensor, target: Tensor = None, *args, **kwargs) -> Tensor:
        # prediction shape: [B, C, H, W]
        """
        Chi tiết hàm: `_compute_loss`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        b, c, h, w = prediction.shape
        count_h = b * c * (h - 1) * w
        count_w = b * c * h * (w - 1)
        
        h_tv = torch.pow((prediction[:, :, 1:, :] - prediction[:, :, :-1, :]), 2).sum()
        w_tv = torch.pow((prediction[:, :, :, 1:] - prediction[:, :, :, :-1]), 2).sum()
        
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w)

    def extra_repr(self) -> str:
        """
        Chi tiết hàm: `extra_repr`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"\n\t weight={self.weight}"
