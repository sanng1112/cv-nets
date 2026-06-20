import argparse
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from layers.base_layer import BaseLayer
from layers.linear_layer import LinearLayer

class FixedMultiHeadAttention(BaseLayer):
    """
    Fixed Multi-Head Attention as proposed by Bhojanapalli et al.
    Decouples head_dim from embed_dim / num_heads to prevent Low-Rank Bottleneck.
    """
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, dropout: float = 0.0, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = LinearLayer(in_features=embed_dim, out_features=self.inner_dim, bias=True)
        self.k_proj = LinearLayer(in_features=embed_dim, out_features=self.inner_dim, bias=True)
        self.v_proj = LinearLayer(in_features=embed_dim, out_features=self.inner_dim, bias=True)
        self.out_proj = LinearLayer(in_features=self.inner_dim, out_features=embed_dim, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return parser

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Chi tiết hàm: `forward`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_dim)
        return self.out_proj(out)
