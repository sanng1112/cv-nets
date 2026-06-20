import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNHead(nn.Module):
    """
    [Chi tiết hàm]: Đầu ra Phân vùng ngữ nghĩa (Fully Convolutional Network Head)
    - Nhận vào Feature Map từ Backbone/Decoder.
    - Rút gọn số kênh và chiếu (project) về số lượng classes.
    - Hỗ trợ nội suy (interpolate) tự động lên kích thước ảnh gốc.
    """
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)
        )
        
    def forward(self, x, target_size=None):
        # x: [B, C, H, W]
        out = self.conv(x)
        
        # Trả về cùng kích thước với nhãn ground truth nếu có yêu cầu
        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
            
        return out
