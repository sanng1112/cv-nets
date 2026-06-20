import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{TanhExp}(x)=x\tanh(e^x)
\]
"""

@register_act_fn(name="tanh_exp")
class TanhExp(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        Chi tiết hàm: `forward`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return x * torch.tanh(torch.exp(x))

    def __repr__(self) -> str:
        """
        Chi tiết hàm: `__repr__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return "TanhExp()"
