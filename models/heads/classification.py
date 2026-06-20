import torch
import torch.nn as nn
from layers.etf_classifier import ETFClassifier

class ClassificationHead(nn.Module):
    """
    [Chi tiết hàm]: Đầu ra Phân loại (Classification Head)
    Hỗ trợ hai chế độ:
    - Linear Classifier truyền thống.
    - Simplex ETF Classifier (Chống Neural Collapse).
    Tự động áp dụng Global Average Pooling nếu đầu vào là Feature Map 2D.
    """
    def __init__(self, in_features: int, num_classes: int, use_etf: bool = False, pool_type: str = "avg", dropout: float = 0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1) if pool_type == "avg" else nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        
        if use_etf:
            self.classifier = ETFClassifier(in_features, num_classes)
        else:
            self.classifier = nn.Linear(in_features, num_classes)
            
    def forward(self, x):
        # Nếu đầu vào là tensor ảnh [B, C, H, W] -> Chuyển thành vector [B, C]
        if x.dim() == 4:
            x = self.pool(x)
            x = self.flatten(x)
            
        x = self.dropout(x)
        return self.classifier(x)
