from typing import Optional
from torch import Tensor, nn

from . import register_act_fn

r'''
\text{ReLU}(x) = \begin{cases}
                x, & \text{if } x > 0 \\                
                0, & \text{otherwise}                
                \end{cases}
'''


@register_act_fn('relu')
class ReLU(nn.ReLU):
    def __init__(self, inplace: bool = False, neg_slope: float = 0.1, *arg, **kwargs):
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__(inplace=inplace)
        self.inplace = inplace 
        
    def __repr__(self):
        """
        Chi tiết hàm: `__repr__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return f'ReLU : inplace { self.inplace }'
    
    

