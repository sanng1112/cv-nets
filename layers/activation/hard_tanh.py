from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{HardTanh}(x)=\mathrm{clip}(x,\min,\max)
\]
"""

@register_act_fn(name="hard_tanh")
class HardTanh(nn.Hardtanh):
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(min_val=min_val, max_val=max_val, inplace=inplace)

    def __repr__(self) -> str:
        """
        Chi tiết hàm: `__repr__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"HardTanh(min_val={self.min_val}, max_val={self.max_val}, inplace={self.inplace})"
