import argparse
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from layers.base_layer import BaseLayer

class ETFClassifier(BaseLayer):
    """
    Simplex Equiangular Tight Frame (ETF) Classifier.
    Proposed based on Papyan et al.'s 'Neural Collapse' discovery.
    The classifier weights are fixed to a Simplex ETF, which is the optimal 
    configuration that networks naturally converge to during the terminal phase of training.
    """
    def __init__(self, in_features: int, num_classes: int, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        
        # ETF requires in_features >= num_classes - 1
        assert in_features >= num_classes - 1, "in_features must be >= num_classes - 1 to form a Simplex ETF"
        
        # Construct the standard Simplex ETF
        # M = sqrt(C / (C - 1)) * (I - 1/C * 1 1^T)
        I = torch.eye(num_classes)
        ones = torch.ones(num_classes, num_classes)
        M = math.sqrt(num_classes / (num_classes - 1)) * (I - (1.0 / num_classes) * ones)
        
        # M is currently [num_classes, num_classes]. We need [in_features, num_classes].
        # We can apply a random orthogonal projection to map it to in_features.
        random_matrix = torch.randn(in_features, num_classes)
        U, _, _ = torch.linalg.svd(random_matrix, full_matrices=False) # U is [in_features, num_classes]
        
        # The ETF weights
        etf_weights = torch.matmul(U, M) # [in_features, num_classes]
        
        # Register as a buffer so it's not trainable, but moves with the model device
        self.register_buffer('weight', etf_weights.t()) # [num_classes, in_features]

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return parser

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, in_features]
        """
        Chi tiết hàm: `forward`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return torch.nn.functional.linear(x, self.weight)
