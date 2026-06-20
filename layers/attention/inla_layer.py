import argparse
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from layers.base_layer import BaseLayer
from layers.linear_layer import LinearLayer

class INLALayer(BaseLayer):
    """
    Inverted Nonlinear Linear Attention (INLA).
    Integrates the theories of Information Bottleneck (Tishby), Rank Collapse (Dong/Oono), 
    and Neural Collapse (Papyan).
    """
    def __init__(self, embed_dim: int, num_heads: int, bottleneck_dim: int, expansion_dim: int, 
                 dropout: float = 0.0, causal: bool = False, *args, **kwargs) -> None:
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
        self.bottleneck_dim = bottleneck_dim
        self.expansion_dim = expansion_dim
        self.causal = causal

        # Step 1: Compression (Information Bottleneck - Tishby)
        self.q_low = LinearLayer(in_features=embed_dim, out_features=bottleneck_dim * num_heads, bias=True)
        self.k_low = LinearLayer(in_features=embed_dim, out_features=bottleneck_dim * num_heads, bias=True)
        
        self.v_proj = LinearLayer(in_features=embed_dim, out_features=embed_dim, bias=True)

        # Step 2: Expansion & Nonlinearity (Rank Collapse Mitigation - Dong/Oono)
        self.act = nn.GELU()
        self.q_exp = LinearLayer(in_features=bottleneck_dim * num_heads, out_features=expansion_dim * num_heads, bias=False)
        self.k_exp = LinearLayer(in_features=bottleneck_dim * num_heads, out_features=expansion_dim * num_heads, bias=False)

        # Step 3: Normalization (Neural Collapse Mitigation - Papyan)
        self.out_norm = nn.LayerNorm(embed_dim)
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

    def feature_map_q(self, x: Tensor) -> Tensor:
        """
        Chi tiết hàm: `feature_map_q`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return self.q_exp(self.act(self.q_low(x)))

    def feature_map_k(self, x: Tensor) -> Tensor:
        """
        Chi tiết hàm: `feature_map_k`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return self.k_exp(self.act(self.k_low(x)))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Chi tiết hàm: `forward`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        B, N, C = x.shape
        
        # Apply Learned Lifting (Phi_INLA)
        q = self.feature_map_q(x).view(B, N, self.num_heads, self.expansion_dim)
        k = self.feature_map_k(x).view(B, N, self.num_heads, self.expansion_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)

        # Ensure positivity for linear attention
        q = torch.nn.functional.elu(q) + 1.0
        k = torch.nn.functional.elu(k) + 1.0

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
        out = self.out_norm(out) # Normalization against ETF collapse
        return self.out_proj(out)
