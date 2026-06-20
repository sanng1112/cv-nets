import argparse
import torch
import torch.nn.functional as F
from torch import Tensor

from loss_fn import LOSS_REGISTRY
from loss_fn.segmentation.base_segmentation import BaseSegmentationCriteria

@LOSS_REGISTRY.register(name="focal_tversky", type="segmentation")
class FocalTverskyLoss(BaseSegmentationCriteria):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(opts, *args, **kwargs)
        self.smooth = getattr(opts, "loss.segmentation.focal_tversky.smooth", 1.0)
        self.alpha = getattr(opts, "loss.segmentation.focal_tversky.alpha", 0.5)
        self.beta = getattr(opts, "loss.segmentation.focal_tversky.beta", 0.5)
        self.gamma = getattr(opts, "loss.segmentation.focal_tversky.gamma", 0.75)
        
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if cls != FocalTverskyLoss:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.segmentation.focal-tversky.smooth",
            type=float,
            default=1.0,
            help="Smoothing factor for Focal Tversky Loss. Defaults to 1.0.",
        )
        group.add_argument(
            "--loss.segmentation.focal-tversky.alpha",
            type=float,
            default=0.5,
            help="Alpha for False Positives penalty. Defaults to 0.5.",
        )
        group.add_argument(
            "--loss.segmentation.focal-tversky.beta",
            type=float,
            default=0.5,
            help="Beta for False Negatives penalty. Defaults to 0.5.",
        )
        group.add_argument(
            "--loss.segmentation.focal-tversky.gamma",
            type=float,
            default=0.75,
            help="Gamma parameter to focus on hard examples. Defaults to 0.75.",
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
        
        TP = (preds * targets).sum()    
        FP = ((1 - targets) * preds).sum()
        FN = (targets * (1 - preds)).sum()
       
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)  
        
        return torch.pow((1 - Tversky), self.gamma)

    def extra_repr(self) -> str:
        """
        Chi tiết hàm: `extra_repr`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"\n\t alpha={self.alpha}\n\t beta={self.beta}\n\t gamma={self.gamma}\n\t smooth={self.smooth}"
