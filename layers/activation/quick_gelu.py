import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{QuickGELU}(x)=x\cdot\sigma(1.702x)
\]
"""

@register_act_fn(name="quick_gelu")
class QuickGELU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        Chi tiết hàm: `forward`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return x * torch.sigmoid(1.702 * x)

    def __repr__(self) -> str:
        """
        Chi tiết hàm: `__repr__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return "QuickGELU()"
