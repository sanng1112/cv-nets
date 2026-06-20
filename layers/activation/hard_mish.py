import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{HardMish}(x)\approx \mathrm{Mish}(x)
\]
"""

@register_act_fn(name="hard_mish")
class HardMish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        Chi tiết hàm: `forward`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return 0.5 * x * torch.clamp(x + 2.0, min=0.0, max=2.0)

    def __repr__(self) -> str:
        """
        Chi tiết hàm: `__repr__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return "HardMish()"
