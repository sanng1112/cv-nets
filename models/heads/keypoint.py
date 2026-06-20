import torch
import torch.nn as nn

class HeatmapKeypointHead(nn.Module):
    """
    [Chi tiết hàm]: Đầu Nhận diện Điểm neo (Pose/Keypoint Estimation Head)
    Sử dụng các lớp Transposed Convolution (Deconv) để phóng to Feature Map,
    nhằm dự đoán Heatmap (Bản đồ nhiệt xác suất) cho các khớp/điểm neo trên cơ thể người hoặc vật thể.
    """
    def __init__(self, in_channels: int, num_keypoints: int, num_deconv_layers: int = 3, num_filters: int = 256):
        super().__init__()
        layers = []
        in_c = in_channels
        
        # Các lớp Deconv để Up-sample không gian ảnh
        for i in range(num_deconv_layers):
            layers.extend([
                nn.ConvTranspose2d(
                    in_channels=in_c,
                    out_channels=num_filters,
                    kernel_size=4,
                    stride=2, # Scale x2 mỗi lớp
                    padding=1,
                    output_padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True)
            ])
            in_c = num_filters
            
        # Lớp chập cuối cùng đẩy ra Heatmap cho từng Keypoint riêng biệt
        layers.append(
            nn.Conv2d(in_channels=in_c, out_channels=num_keypoints, kernel_size=1, stride=1, padding=0)
        )
        
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        # Đầu ra ma trận [B, num_keypoints, H_out, W_out]
        return self.head(x)
