import torch
import torch.nn as nn

class DecoupledDetectionHead(nn.Module):
    """
    [Chi tiết hàm]: Đầu ra Nhận diện Vật thể Phân mảnh (Decoupled Detection Head)
    Lấy cảm hứng từ YOLOX và FCOS. Tách biệt hoàn toàn nhánh Phân loại (Classification)
    và nhánh Hồi quy tọa độ (Regression) để tránh xung đột không gian đặc trưng.
    """
    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 256):
        super().__init__()
        
        # Nén kênh để giảm độ phức tạp tính toán
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )
        
        # Nhánh phân loại (Đoán xem vật thể là con mèo hay cái bàn)
        self.cls_branch = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, 1) # Output: Logits cho C classes
        )
        
        # Nhánh hồi quy (Đoán Box: tọa độ x, y và kích thước w, h)
        self.reg_branch = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 4, 1) # Output: 4 giá trị bounding box
        )
        
        # Nhánh đánh giá chất lượng Box (Centerness hoặc Objectness score)
        self.obj_branch = nn.Conv2d(hidden_channels, 1, 1)
        
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.stem(x)
        
        cls_out = self.cls_branch(x)    # [B, num_classes, H, W]
        reg_out = self.reg_branch(x)    # [B, 4, H, W]
        obj_out = self.obj_branch(x)    # [B, 1, H, W]
        
        return {
            "cls": cls_out,
            "reg": reg_out,
            "obj": obj_out
        }
