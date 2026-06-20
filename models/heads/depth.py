import torch
import torch.nn as nn

class DepthEstimationHead(nn.Module):
    """
    [Chi tiết hàm]: Đầu Ước lượng Chiều sâu (Monocular Depth Head)
    Đầu ra là một bản đồ độ sâu (Depth Map) đơn kênh. Dùng hàm Sigmoid (nếu độ sâu đã chuẩn hóa 0-1)
    hoặc ReLU để đảm bảo khoảng cách chiều sâu luôn mang giá trị dương.
    """
    def __init__(self, in_channels: int, use_sigmoid: bool = True):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )
        self.use_sigmoid = use_sigmoid
        
    def forward(self, x):
        # x: [B, C, H, W]
        out = self.conv_block(x)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        else:
            out = torch.relu(out)
        return out # [B, 1, H, W]
