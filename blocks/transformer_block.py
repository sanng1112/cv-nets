import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from layers.attention.multihead_attention import MultiHeadAttention
from layers.attention.fixed_multihead_attention import FixedMultiHeadAttention
from layers.attention.linear_attention import LinearAttention
from layers.attention.inla_layer import INLALayer
from layers.linear_layer import LinearLayer

class TransformerBlock(nn.Module):
    """
    Standard Transformer Block.
    Dong et al. mathematically proved that the Skip Connections and the MLPs here 
    are essential to prevent Rank Collapse in the Attention mechanism.
    Can be configured to use 'multihead', 'fixed_multihead', or 'linear' attention.
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 attention_type: str = "multihead", head_dim: Optional[int] = None, 
                 bottleneck_dim: Optional[int] = None, expansion_dim: Optional[int] = None, causal: bool = False):
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        if attention_type == "multihead":
            self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        elif attention_type == "fixed_multihead":
            assert head_dim is not None, "head_dim is required for fixed_multihead"
            self.attn = FixedMultiHeadAttention(embed_dim, num_heads, head_dim, dropout)
        elif attention_type == "linear":
            self.attn = LinearAttention(embed_dim, num_heads, dropout, causal=causal)
        elif attention_type == "inla":
            assert bottleneck_dim is not None and expansion_dim is not None, "INLA requires bottleneck_dim and expansion_dim"
            self.attn = INLALayer(embed_dim, num_heads, bottleneck_dim, expansion_dim, dropout, causal=causal)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
            
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        # The MLP serves as a high-Lipschitz non-linear mapping to prevent rank collapse (Dong et al. 2021)
        self.mlp = nn.Sequential(
            LinearLayer(embed_dim, mlp_hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            LinearLayer(mlp_hidden_dim, embed_dim, bias=True),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Skip connection 1 prevents initial rank collapse
        """
        Chi tiết hàm: `forward`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        x = x + self.attn(self.norm1(x), mask=mask)
        # Skip connection 2 + MLP increases representation diversity
        x = x + self.mlp(self.norm2(x))
        return x
