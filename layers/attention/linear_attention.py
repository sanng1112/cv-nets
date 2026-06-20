import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from layers.base_layer import BaseLayer
from layers.linear_layer import LinearLayer

class LinearAttention(BaseLayer):
    """
    Linear Attention as proposed by Katharopoulos et al.
    Reduces complexity to O(N) by using associative property and replacing Softmax with a kernel feature map.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, causal: bool = False, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        self.q_proj = LinearLayer(in_features=embed_dim, out_features=embed_dim, bias=True)
        self.k_proj = LinearLayer(in_features=embed_dim, out_features=embed_dim, bias=True)
        self.v_proj = LinearLayer(in_features=embed_dim, out_features=embed_dim, bias=True)
        self.out_proj = LinearLayer(in_features=embed_dim, out_features=embed_dim, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return parser

    def feature_map(self, x: Tensor) -> Tensor:
        """
        Chi tiết hàm: `feature_map`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return F.elu(x) + 1.0

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Chi tiết hàm: `forward`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        B, N, C = x.shape
        q = self.feature_map(self.q_proj(x).view(B, N, self.num_heads, self.head_dim))
        k = self.feature_map(self.k_proj(x).view(B, N, self.num_heads, self.head_dim))
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)

        if self.causal:
            kv = torch.einsum('bnhd,bnhm->bnhdm', k, v)
            kv_cumsum = torch.cumsum(kv, dim=1) 
            
            k_cumsum = torch.cumsum(k, dim=1) 
            
            num = torch.einsum('bnhd,bnhdm->bnhm', q, kv_cumsum)
            den = torch.einsum('bnhd,bnhd->bnh', q, k_cumsum).unsqueeze(-1) + 1e-6
            out = num / den
        else:
            kv = torch.einsum('bnhd,bnhm->bhdm', k, v)
            num = torch.einsum('bnhd,bhdm->bnhm', q, kv)
            den = torch.einsum('bnhd,bhd->bnh', q, k.sum(dim=1)).unsqueeze(-1) + 1e-6
            out = num / den
            
        out = out.contiguous().view(B, N, C)
        return self.out_proj(out)
