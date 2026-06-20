from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{HardSwish}(x)=x\cdot\frac{\mathrm{ReLU6}(x+3)}{6}
\]
"""

@register_act_fn(name="hard_swish")
class HardSwish(nn.Hardswish):
    def __init__(self, inplace: bool = False) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(inplace=inplace)

    def __repr__(self) -> str:
        """
        Chi tiết hàm: `__repr__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"HardSwish(inplace={self.inplace})"
