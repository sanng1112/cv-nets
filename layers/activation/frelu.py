import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{FReLU}(x)=\max(x,\mathrm{BN}(\mathrm{DWConv}(x)))
\]
"""

@register_act_fn(name="frelu")
class FReLU(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__()
        padding = kernel_size // 2
        self.dw_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.channels = channels
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        """
        Chi tiết hàm: `forward`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return torch.max(x, self.bn(self.dw_conv(x)))

    def __repr__(self) -> str:
        """
        Chi tiết hàm: `__repr__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"FReLU(channels={self.channels}, kernel_size={self.kernel_size})"
