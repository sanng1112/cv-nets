import argparse
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from layers.base_layer import BaseLayer

class SpectralScaledGCN(BaseLayer):
    """
    Graph Convolutional Network Layer with Spectral Singular Value Scaling.
    Proposed by Oono & Suzuki to prevent Oversmoothing / Information Loss.
    The maximum singular value of the weight matrix is controlled/scaled to a target value 's'.
    """
    def __init__(self, in_channels: int, out_channels: int, target_singular_value: float = 1.0, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_s = target_singular_value
        
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Chi tiết hàm: `reset_parameters`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return parser

    def forward(self, x: Tensor, adj_matrix: Tensor) -> Tensor:
        """
        x: Node features [B, N, C] or [N, C]
        adj_matrix: Normalized adjacency matrix (augmented) [N, N]
        """
        # During training/forward, we scale the weight matrix's max singular value
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(self.weight, full_matrices=False)
            max_s = S.max()
            if max_s > self.target_s:
                # Scale the weight so its max singular value is target_s
                self.weight.data = self.weight.data * (self.target_s / max_s)
        
        # Support both [N, C] and [B, N, C]
        if x.dim() == 2:
            support = torch.matmul(x, self.weight)
            out = torch.matmul(adj_matrix, support)
        else:
            support = torch.matmul(x, self.weight)
            out = torch.matmul(adj_matrix.unsqueeze(0), support)
            
        return out + self.bias
