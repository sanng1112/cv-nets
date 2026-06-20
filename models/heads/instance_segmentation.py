import torch
import torch.nn as nn

class ProtoMaskHead(nn.Module):
    """
    [Chi tiết hàm]: Đầu Phân vùng Thực thể (Instance Segmentation Proto-Head)
    Dựa trên ý tưởng của YOLACT / YOLOv8-Seg. Mạng này gọi là ProtoNet:
    Nhiệm vụ của nó là sinh ra tập hợp các Prototype Masks (Mask mẫu cơ sở chung cho toàn bộ ảnh).
    Sau đó, nhánh Regression của mạng Detection sẽ xuất ra các "Coefficients" (hệ số) để nhân vào các Mask mẫu này nhằm tách ra từng vật thể.
    """
    def __init__(self, in_channels: int, proto_channels: int = 256, num_prototypes: int = 32):
        super().__init__()
        self.protonet = nn.Sequential(
            nn.Conv2d(in_channels, proto_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(proto_channels),
            nn.ReLU(inplace=True),
            
            # Phóng to Mask (Up-sample) để đạt độ nét cao hơn
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(proto_channels, proto_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(proto_channels),
            nn.ReLU(inplace=True),
            
            # Xuất ra Prototype Masks
            nn.Conv2d(proto_channels, num_prototypes, 1),
            nn.ReLU(inplace=True) # Hàm kích hoạt để mask luôn mang giá trị dương
        )

    def forward(self, x):
        # Output: [B, num_prototypes, H*2, W*2]
        return self.protonet(x)
