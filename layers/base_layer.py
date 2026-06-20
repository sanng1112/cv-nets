import argparse

from typing import Any, Dict, Optional, List, Literal, Tuple
from torch import Tensor, nn


class BaseLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return parser

    def forward(self, *args, **kwargs) -> Any:
        """
        Chi tiết hàm: `forward`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        raise NotImplementedError
    
    # Khởi tạo trọng số cho lớp
    def int_weight(self) -> None:
        """
        Chi tiết hàm: `int_weight`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        pass
        
    def visualize_weight_distribution(self, layer_name: Optional[str] = None) -> None:
        """Trực quan hóa phân phối trọng số của layer"""
        from utils.visualization import visualize_weight_distribution
        visualize_weight_distribution(self, layer_name)

    def visualize_weight_heatmap(self, layer_name: Optional[str] = None) -> None:
        """Trực quan hóa heatmap của trọng số (tốt cho phân tích Conv/Linear)"""
        from utils.visualization import visualize_weight_heatmap
        visualize_weight_heatmap(self, layer_name)

    def register_feature_map_hook(self) -> Any:
        """Đăng ký hook để lấy feature map sau forward pass, hỗ trợ phân tích"""
        from utils.visualization import FeatureMapHook
        return FeatureMapHook(self)
    