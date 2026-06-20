from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{RReLU}(x)=
\begin{cases}
x, & x>0 \\
\alpha x, & x\le 0,\ \alpha\sim \mathcal{U}(l,u)
\end{cases}
\]
"""

@register_act_fn(name="rrelu")
class RReLU(nn.RReLU):
    def __init__(self, lower: float = 1.0 / 8.0, upper: float = 1.0 / 3.0, inplace: bool = False, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(lower=lower, upper=upper, inplace=inplace)

    def __repr__(self) -> str:
        """
        Chi tiết hàm: `__repr__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"RReLU(lower={self.lower}, upper={self.upper}, inplace={self.inplace})"
