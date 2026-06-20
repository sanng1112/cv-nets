import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import register_act_fn


r"""
\[
\mathrm{StarReLU}(x)=s\cdot \mathrm{ReLU}(x)^2+b
\]
"""

@register_act_fn(name="star_relu")
class StarReLU(nn.Module):
    def __init__(self, scale: float = 1.0, bias: float = 0.0, inplace: bool = False) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__()
        self.scale = scale
        self.bias = bias
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        """
        Chi tiết hàm: `forward`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        y = F.relu(x, inplace=self.inplace)
        return self.scale * (y * y) + self.bias

    def __repr__(self) -> str:
        """
        Chi tiết hàm: `__repr__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"StarReLU(scale={self.scale}, bias={self.bias}, inplace={self.inplace})"
