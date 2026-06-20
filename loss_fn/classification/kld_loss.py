import argparse
import torch
from torch import Tensor
from torch.nn import functional as F

from loss_fn import LOSS_REGISTRY
from loss_fn.classification.base_classification import BaseClassificationCriteria

@LOSS_REGISTRY.register(name="kld", type="classification")
class KLDivergenceLoss(BaseClassificationCriteria):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(opts, *args, **kwargs)
        self.log_target = getattr(opts, "loss.classification.kld.log_target", False)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if cls != KLDivergenceLoss:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.classification.kld.log-target",
            action="store_true",
            default=False,
            help="If True, target is assumed to be log probabilities.",
        )
        return parser    

    def _compute_loss(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        # Prediction needs to be log probabilities for KLDivLoss
        """
        Chi tiết hàm: `_compute_loss`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        prediction_log_prob = F.log_softmax(prediction, dim=-1)
        if target.shape != prediction.shape:
            # Assuming target are indices, convert to one-hot probability distribution
            target = F.one_hot(target, num_classes=prediction.shape[-1]).float()
        
        return F.kl_div(prediction_log_prob, target, reduction='batchmean', log_target=self.log_target)

    def extra_repr(self) -> str:
        """
        Chi tiết hàm: `extra_repr`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"\n\t log_target={self.log_target}"
