import argparse
import torch
import torch.nn.functional as F
from torch import Tensor

from loss_fn import LOSS_REGISTRY
from loss_fn.segmentation.base_segmentation import BaseSegmentationCriteria

@LOSS_REGISTRY.register(name="jaccard", type="segmentation")
class JaccardLoss(BaseSegmentationCriteria):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(opts, *args, **kwargs)
        self.smooth = getattr(opts, "loss.segmentation.jaccard.smooth", 1.0)
        
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if cls != JaccardLoss:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.segmentation.jaccard.smooth",
            type=float,
            default=1.0,
            help="Smoothing factor for Jaccard (IoU) Loss. Defaults to 1.0.",
        )
        return parser    

    def _compute_loss(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """
        Chi tiết hàm: `_compute_loss`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        preds = torch.softmax(prediction, dim=1)
        if target.dim() == prediction.dim() - 1:
            target = F.one_hot(target, num_classes=prediction.shape[1]).permute(0, 3, 1, 2).float()
            
        preds = preds.contiguous().view(-1)
        targets = target.contiguous().view(-1)
        
        intersection = (preds * targets).sum()
        total = (preds + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - IoU

    def extra_repr(self) -> str:
        """
        Chi tiết hàm: `extra_repr`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"\n\t smooth={self.smooth}"
