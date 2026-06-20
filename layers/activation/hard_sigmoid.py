from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{HardSigmoid}(x)\approx \sigma(x)
\]
"""

@register_act_fn(name="hard_sigmoid")
class HardSigmoid(nn.Hardsigmoid):
    def __init__(self, inplace: bool = False) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__()
        self.inplace = inplace

    def __repr__(self) -> str:
        """
        Chi tiết hàm: `__repr__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"HardSigmoid(inplace={self.inplace})"
