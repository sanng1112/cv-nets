from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{ReLU6}(x)=\min(\max(0,x),6)
\]
"""

@register_act_fn(name="relu6")
class ReLU6(nn.ReLU6):
    def __init__(self, inplace: bool = False, *args, **kwargs) -> None:
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
        return f"ReLU6(inplace={self.inplace})"
