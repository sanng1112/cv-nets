"""Multi-Head Self-Attention — scaled dot-product attention with QKV projection."""

from math import sqrt
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MultiHeadSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention.

    Projects input into queries, keys, values via a single linear layer,
    splits into multiple heads, computes scaled dot-product attention,
    and projects back to ``embed_dim``.

    Parameters
    ----------
    embed_dim : int
        Total embedding dimension (per token).
    num_heads : int
        Number of attention heads.  Must divide ``embed_dim`` evenly.
    dropout : float
        Dropout probability applied to attention weights (default ``0.0``).
    bias : bool
        Whether to include bias in QKV and output projections (default ``False``).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, causal_mask: bool = False) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / self.scale

        if causal_mask:
            mask = torch.ones(N, N, device=x.device, dtype=torch.bool).triu(diagonal=1)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}"
        )
