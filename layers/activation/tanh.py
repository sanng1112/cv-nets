from . import register_act_fn
from torch import nn


r"""
\[
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
\]
"""

@register_act_fn(name="tanh")
class Tanh(nn.Tanh):
    def __init__(self, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__()

    def __repr__(self) -> str:
        """
        Chi tiết hàm: `__repr__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return "Tanh()"
