"""TransformerMLP — the two-layer MLP used in transformer blocks."""

from typing import Any

from torch import Tensor, nn

from cvnets.core.base_block import BaseBlock
from cvnets.core.registry import BLOCK_REGISTRY


@BLOCK_REGISTRY.register("transformer_mlp")
@BLOCK_REGISTRY.register("TransformerMLP")
class TransformerMLP(BaseBlock):
    """Two-layer MLP with GELU activation, as used in transformer blocks.

    Structure: ``Linear(embed_dim, hidden_dim) → GELU → Dropout →
    Linear(hidden_dim, embed_dim) → Dropout``

    Parameters
    ----------
    embed_dim : int
        Input/output embedding dimension.
    expansion_ratio : int or float
        Hidden dimension multiplier (default ``4``).
    dropout : float
        Dropout probability after each linear layer (default ``0.0``).
    **kwargs
        Extra keyword arguments (ignored; accepted for config compatibility).
    """

    def __init__(
        self,
        embed_dim: int,
        expansion_ratio: float = 4.0,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * expansion_ratio)
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

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.fc1.in_features}, "
            f"hidden_dim={self.fc1.out_features}"
        )
