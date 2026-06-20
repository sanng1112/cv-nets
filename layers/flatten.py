from typing import Optional

from torch import Tensor, nn


class Flatten(nn.Flatten):
    def __init__(self, start_dim: Optional[int] = 1, end_dim: Optional[int] = -1):
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super(Flatten, self).__init__(start_dim=start_dim, end_dim=end_dim)
