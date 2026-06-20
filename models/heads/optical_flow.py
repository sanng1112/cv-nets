import torch
import torch.nn as nn

class OpticalFlowHead(nn.Module):
    """
    [Chi tiết hàm]: Đầu Phân tích Chuyển động (Optical Flow Head)
    Dự đoán luồng quang học (Optical Flow) gồm 2 kênh: delta_x và delta_y 
    để mô tả vector sự dịch chuyển của từng pixel giữa 2 khung hình liên tiếp.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels // 2, 2, kernel_size=3, padding=1) # 2 channels for dx, dy
        )
        
    def forward(self, x):
        return self.conv_block(x) # [B, 2, H, W]
