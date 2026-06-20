from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{GELU}(x)=x\Phi(x)
\]
"""

@register_act_fn(name="gelu")
class GELU(nn.GELU):
    def __init__(self, approximate: str = "none") -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(approximate=approximate)

    def __repr__(self) -> str:
        """
        Chi tiết hàm: `__repr__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"GELU(approximate={self.approximate})"
