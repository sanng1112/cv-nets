import torch
import torch.nn as nn

class PixelShuffleHead(nn.Module):
    """
    [Chi tiết hàm]: Đầu Phục hồi Ảnh / Siêu phân giải (Super Resolution / Restoration Head)
    Sử dụng kỹ thuật Sub-Pixel Convolution (PixelShuffle) để phóng to ảnh mà không gây nhiễu bàn cờ (Checkerboard artifacts) như Deconv thông thường.
    """
    def __init__(self, in_channels: int, out_channels: int = 3, upscale_factor: int = 2):
        super().__init__()
        # Nhân số kênh lên theo bình phương tỷ lệ phóng to
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        # x: [B, in_channels, H, W]
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        return out # [B, out_channels, H * upscale_factor, W * upscale_factor]
