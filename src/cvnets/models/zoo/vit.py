"""
Vision Transformer (ViT) model zoo — Tiny, Small, Base.

Factory functions return a plain ``nn.Module`` and are registered in
``MODEL_REGISTRY``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from cvnets.core.registry import MODEL_REGISTRY
from cvnets.layers.patch_embedding import PatchEmbedding
from cvnets.layers.positional_encoding import LearnedPositionalEncoding
from cvnets.blocks.transformer_block import TransformerEncoderBlock
from cvnets.layers.linear_layer import LinearLayer
from cvnets.utils.logger import info


# ===================================================================
# _ViT
# ===================================================================


class _ViT(nn.Module):
    """Vision Transformer backbone (Dosovitskiy et al. 2020).

    Parameters
    ----------
    embed_dim : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder blocks.
    patch_size : int
        Patch size (default 16).
    image_size : int
        Input image resolution (default 224).
    in_channels : int
        Number of input channels (default 3).
    num_classes : int
        Number of output classes (default 1000).
    mlp_ratio : float
        Hidden-dim to embed-dim ratio in the MLP (default 4.0).
    dropout : float
        Dropout probability (default 0.0).
    drop_path : float
        Stochastic depth rate per block (default 0.0).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        patch_size: int = 16,
        image_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 1000,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # ---- Patch embedding ----
        self.patch_embed = PatchEmbedding(
            img_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # ---- Class token ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ---- Positional encoding (+1 for class token) ----
        self.pos_embed = LearnedPositionalEncoding(
            num_tokens=num_patches + 1, embed_dim=embed_dim
        )

        # ---- Dropout after positional embedding ----
        self.pos_drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # ---- Transformer encoder blocks ----
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_path,
                )
            )

        # ---- Final norm + classifier ----
        self.norm = nn.LayerNorm(embed_dim)
        self.head = LinearLayer(embed_dim, num_classes, bias=True)

        # Log
        num_params = sum(p.numel() for p in self.parameters())
        info(
            f"_ViT(embed_dim={embed_dim}, heads={num_heads}, "
            f"layers={num_layers}) — {num_params:,} parameters"
        )

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, C)

        # Prepend class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, N+1, C)

        # Add positional encoding
        x = self.pos_embed(x)
        x = self.pos_drop(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Take class token output
        x = self.norm(x)
        x = x[:, 0]  # (B, C)

        # Classifier head
        x = self.head(x)
        return x


# ===================================================================
# Factory functions
# ===================================================================


@MODEL_REGISTRY.register("vit_tiny")
def make_vit_tiny(
    patch_size: int = 16,
    image_size: int = 224,
    num_classes: int = 1000,
) -> _ViT:
    """Build a ViT-Tiny (embed_dim=192, heads=12, layers=12)."""
    return _ViT(
        embed_dim=192,
        num_heads=12,
        num_layers=12,
        patch_size=patch_size,
        image_size=image_size,
        num_classes=num_classes,
    )


@MODEL_REGISTRY.register("vit_small")
def make_vit_small(
    patch_size: int = 16,
    image_size: int = 224,
    num_classes: int = 1000,
) -> _ViT:
    """Build a ViT-Small (embed_dim=384, heads=6, layers=12)."""
    return _ViT(
        embed_dim=384,
        num_heads=6,
        num_layers=12,
        patch_size=patch_size,
        image_size=image_size,
        num_classes=num_classes,
    )


@MODEL_REGISTRY.register("vit_base")
def make_vit_base(
    patch_size: int = 16,
    image_size: int = 224,
    num_classes: int = 1000,
) -> _ViT:
    """Build a ViT-Base (embed_dim=768, heads=12, layers=12)."""
    return _ViT(
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        patch_size=patch_size,
        image_size=image_size,
        num_classes=num_classes,
    )
