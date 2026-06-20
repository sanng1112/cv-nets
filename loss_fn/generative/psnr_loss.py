import argparse
import torch
from torch import Tensor

from loss_fn import LOSS_REGISTRY
from loss_fn.generative.base_generative import BaseGenerativeCriteria

@LOSS_REGISTRY.register(name="psnr", type="generative")
class PSNRLoss(BaseGenerativeCriteria):
    """
    Peak Signal-to-Noise Ratio (PSNR) Loss.
    Often used in image restoration (super-resolution, denoising).
    Since we want to maximize PSNR, the loss minimizes negative PSNR.
    """
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(opts, *args, **kwargs)
        self.max_val = getattr(opts, "loss.generative.psnr.max_val", 1.0)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if cls != PSNRLoss:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.generative.psnr.max-val",
            type=float,
            default=1.0,
            help="Maximum value of the images (e.g., 1.0 or 255.0).",
        )
        return parser

    def _compute_loss(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """
        Chi tiết hàm: `_compute_loss`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        mse = torch.mean((prediction - target) ** 2)
        if mse == 0:
            return torch.tensor(0.0, device=prediction.device)
        psnr = 10 * torch.log10((self.max_val ** 2) / mse)
        # Minimize negative PSNR
        return -psnr

    def extra_repr(self) -> str:
        """
        Chi tiết hàm: `extra_repr`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"\n\t max_val={self.max_val}"
