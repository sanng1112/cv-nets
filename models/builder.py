import timm
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from models.heads import (
    ClassificationHead, 
    DecoupledDetectionHead, 
    FCNHead,
    ArcFaceHead,
    HeatmapKeypointHead,
    ProtoMaskHead,
    PixelShuffleHead,
    DepthEstimationHead,
    OpticalFlowHead
)

class CVNetModel(nn.Module):
    """
    [Chi tiết hàm]: Trình Xây dựng Mô hình (Model Builder)
    Sử dụng Backbone từ thư viện `timm` (với hơn 1000+ mô hình SOTA).
    Tự động cắt bỏ Classifier cũ của tác giả và nối vào các Heads tùy chỉnh của cv-nets.
    """
    def __init__(self, backbone_name: str, head_type: str, head_kwargs: Dict[str, Any] = None, pretrained: bool = True, **backbone_kwargs):
        super().__init__()
        if head_kwargs is None:
            head_kwargs = {}
            
        # 1. Tải Backbone từ timm (Cắt bỏ head cũ bằng num_classes=0)
        try:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='', **backbone_kwargs)
        except Exception as e:
            raise ValueError(f"Không tìm thấy Backbone '{backbone_name}' trong timm. Lỗi: {e}")
            
        # Lấy số kênh (channels) đầu ra tự động
        if hasattr(self.backbone, 'num_features'):
            in_channels = self.backbone.num_features
        elif hasattr(self.backbone, 'embed_dim'):
            in_channels = self.backbone.embed_dim
        else:
            # Fake một lần chạy để lấy số chiều kênh
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_out = self.backbone(dummy_input)
            in_channels = dummy_out.shape[1]
            
        # 2. Ráp nối Head chuyên dụng
        if head_type == "classification":
            self.head = ClassificationHead(in_features=in_channels, **head_kwargs)
        elif head_type == "detection":
            self.head = DecoupledDetectionHead(in_channels=in_channels, **head_kwargs)
        elif head_type == "segmentation":
            self.head = FCNHead(in_channels=in_channels, **head_kwargs)
        elif head_type == "metric_learning":
            self.head = ArcFaceHead(in_features=in_channels, **head_kwargs)
        elif head_type == "keypoint":
            self.head = HeatmapKeypointHead(in_channels=in_channels, **head_kwargs)
        elif head_type == "instance_segmentation":
            self.head = ProtoMaskHead(in_channels=in_channels, **head_kwargs)
        elif head_type == "super_resolution":
            self.head = PixelShuffleHead(in_channels=in_channels, **head_kwargs)
        elif head_type == "depth":
            self.head = DepthEstimationHead(in_channels=in_channels, **head_kwargs)
        elif head_type == "optical_flow":
            self.head = OpticalFlowHead(in_channels=in_channels)
        else:
            raise NotImplementedError(f"Head '{head_type}' chưa được hỗ trợ.")

    def forward(self, x, **kwargs):
        # Đặc trưng được nhai qua Backbone siêu mạnh
        features = self.backbone(x)
        # Đưa vào Head xử lý chuyên biệt
        return self.head(features, **kwargs)

    @classmethod
    def build_from_yaml(cls, opts: Any) -> "CVNetModel":
        """
        [Trạm Lắp Ráp Tự Động]: Đọc file cấu hình YAML (đã chuyển thành opts)
        và tự động gọi các linh kiện (Backbone, Head, Channels) tương ứng.
        """
        backbone_name = getattr(opts, "model_name", "resnet18")
        head_type = getattr(opts, "task_type", "classification")
        pretrained = getattr(opts, "pretrained", True)
        
        # Bóc tách cấu hình riêng biệt cho từng Head
        head_kwargs = {}
        if head_type in ["classification", "segmentation", "detection", "metric_learning"]:
            head_kwargs["num_classes"] = getattr(opts, "num_classes", 10)
            
        print(f"[Model Builder] Tự động khởi tạo: {backbone_name} + {head_type} (Classes: {head_kwargs.get('num_classes', 'N/A')})")
        return cls(
            backbone_name=backbone_name, 
            head_type=head_type, 
            head_kwargs=head_kwargs, 
            pretrained=pretrained,
            in_chans=getattr(opts, "in_chans", 3)
        )
