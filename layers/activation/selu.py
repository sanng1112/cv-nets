from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{SELU}(x)=\lambda
\begin{cases}
x, & x>0 \\
\alpha(e^x-1), & x\le 0
\end{cases}
\]
"""

@register_act_fn(name="selu")
class SELU(nn.SELU):
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
        return f"SELU(inplace={self.inplace})"
