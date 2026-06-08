"""Transformer Encoder Block — pre-norm MHSA + MLP with residual connections."""

from typing import Any

from torch import Tensor, nn

from cvnets.core.base_block import BaseBlock
from cvnets.core.registry import BLOCK_REGISTRY
from cvnets.layers.drop_path import DropPath
from cvnets.layers.layer_scale import LayerScale
from cvnets.layers.multi_head_attention import MultiHeadSelfAttention


@BLOCK_REGISTRY.register("transformer_encoder")
@BLOCK_REGISTRY.register("TransformerEncoderBlock")
class TransformerEncoderBlock(BaseBlock):
    """Transformer encoder block with pre-normalisation.

    Structure::

        x → x + DropPath(LayerScale(Attention(Norm(x))))
          → x + DropPath(LayerScale(  MLP  (Norm(x))))

    Parameters
    ----------
    embed_dim : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        Hidden/embedding ratio for the MLP (default ``4.0``).
    dropout : float
        Dropout rate for attention and MLP projections (default ``0.0``).
    drop_path : float
        Stochastic depth rate for both residual branches (default ``0.0``).
    layer_scale_init : float
        If ``> 0``, apply LayerScale after each sub-layer with this init
        value (default ``0.0`` = no LayerScale).
    **kwargs
        Extra keyword arguments (ignored; accepted for config compatibility).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.ls1 = (
            LayerScale(embed_dim, init_value=layer_scale_init)
            if layer_scale_init > 0
            else nn.Identity()
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _TransformerMLPInner(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.ls2 = (
            LayerScale(embed_dim, init_value=layer_scale_init)
            if layer_scale_init > 0
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.norm1.normalized_shape[0]}, "
            f"num_heads={self.attn.num_heads}"
        )


class _TransformerMLPInner(nn.Module):
    """Internal MLP used inside TransformerEncoderBlock (not registered)."""

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
