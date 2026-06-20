from torch import Tensor, nn
from torch.nn import functional as F

from . import register_act_fn


r"""
\[
f(x)=
\begin{cases}
x, & x>\theta \\
0, & \text{otherwise}
\end{cases}
\]
"""

@register_act_fn(name="threshold_relu")
class ThresholdReLU(nn.Module):
    def __init__(self, threshold: float = 1.0, value: float = 0.0) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__()
        self.threshold = threshold
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        """
        Chi tiết hàm: `forward`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return F.threshold(x, self.threshold, self.value)

    def __repr__(self) -> str:
        """
        Chi tiết hàm: `__repr__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"ThresholdReLU(threshold={self.threshold}, value={self.value})"
