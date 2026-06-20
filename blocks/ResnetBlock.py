import copy
import torch
from torch import nn, Tensor
from typing import Optional, Any
from layers import build_activation_layer
from utils.config_helper import get_param
from blocks.ConvBNAct import ConvBNAct
class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        opts: Optional[Any] = None
    ):
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__()
        
        # 1. Nhánh Conv 1: Có Activation (Giữ nguyên opts)
        self.conv1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            opts=opts
        )
        
        # Tạo opts mới không có activation cho conv2 và shortcut
        opts_no_act = self._remove_act_from_opts(opts)
        
        # 2. Nhánh Conv 2: KHÔNG CÓ Activation
        self.conv2 = ConvBNAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            opts=opts_no_act
        )
        
        # 3. Kết nối tắt (Shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = ConvBNAct(
                in_channels=in_channels,
                out_channels=out_channels * self.expansion,
                kernel_size=1,
                stride=stride,
                padding=0,        
                opts=opts_no_act  # Shortcut cũng KHÔNG CÓ Activation
            )
            
        # 4. Final Activation
        self.final_act = build_activation_layer(opts=get_param(opts, None, 'act', None))

    def _remove_act_from_opts(self, opts: Any) -> Any:
        """
        Phiên bản nâng cấp: Xử lý an toàn cho cả Dict và SimpleNamespace
        để đảm bảo conv2 và shortcut hoàn toàn không sinh ra lớp act.
        """
        if opts is None:
            return None
            
        opts_copy = copy.deepcopy(opts)
        
        # Xử lý nếu là Dictionary
        if isinstance(opts_copy, dict) and 'act' in opts_copy:
            opts_copy['act'] = None
            
        # Xử lý nếu là SimpleNamespace (do hàm dict_to_namespace của bạn tạo ra)
        elif hasattr(opts_copy, 'act'):
            setattr(opts_copy, 'act', None)
            
        return opts_copy

    def forward(self, x: Tensor) -> Tensor:
        """
        Chi tiết hàm: `forward`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        # SỬA LỖI TẠI ĐÂY: Thay `out += identity` bằng `out = out + identity`
        # Việc này tạo ra một Tensor mới, không ghi đè lên giá trị gốc,
        # giúp quá trình backward pass (tính đạo hàm) không bị crash.
        out = out + identity
        
        if self.final_act is not None:
            out = self.final_act(out)
            
        return out