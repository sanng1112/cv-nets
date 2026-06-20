import argparse
import torch
import torch.nn.functional as F
from torch import Tensor
import math

from loss_fn import LOSS_REGISTRY
from loss_fn.generative.base_generative import BaseGenerativeCriteria

def gaussian(window_size, sigma):
    """
    Chi tiết hàm: `gaussian`
    - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
    - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
    """
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    """
    Chi tiết hàm: `create_window`
    - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
    - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    Chi tiết hàm: `_ssim`
    - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
    - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
    """
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

@LOSS_REGISTRY.register(name="ssim", type="generative")
class SSIMLoss(BaseGenerativeCriteria):
    """
    Structural Similarity (SSIM) Loss.
    Maximizing SSIM is equivalent to minimizing (1 - SSIM).
    """
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(opts, *args, **kwargs)
        self.window_size = getattr(opts, "loss.generative.ssim.window_size", 11)
        self.size_average = True
        self.channel = 3
        self.window = create_window(self.window_size, self.channel)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if cls != SSIMLoss:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.generative.ssim.window-size",
            type=int,
            default=11,
            help="Window size for SSIM. Defaults to 11.",
        )
        return parser

    def _compute_loss(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """
        Chi tiết hàm: `_compute_loss`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        if prediction.shape[1] != self.channel:
            self.channel = prediction.shape[1]
            self.window = create_window(self.window_size, self.channel).to(prediction.device)
        
        if self.window.device != prediction.device:
            self.window = self.window.to(prediction.device)
            
        ssim = _ssim(prediction, target, self.window, self.window_size, self.channel, self.size_average)
        return 1 - ssim

    def extra_repr(self) -> str:
        """
        Chi tiết hàm: `extra_repr`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"\n\t window_size={self.window_size}"
