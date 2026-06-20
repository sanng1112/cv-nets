import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{GLU}(x)=a\odot\sigma(b), \quad [a,b]=\mathrm{chunk}(x)
\]
"""

@register_act_fn(name="glu")
class GLU(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Chi tiết hàm: `forward`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        a, b = torch.chunk(x, 2, dim=self.dim)
        return a * torch.sigmoid(b)

    def __repr__(self) -> str:
        """
        Chi tiết hàm: `__repr__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f"GLU(dim={self.dim})"
