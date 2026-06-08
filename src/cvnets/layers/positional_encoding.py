"""Positional Encoding — sinusoidal and learned variants for transformer models."""

import math

import torch
from torch import Tensor, nn


def sinusoidal_positional_encoding(
    num_tokens: int,
    embed_dim: int,
) -> Tensor:
    """Generate sinusoidal positional encodings (Vaswani et al., 2017).

    Parameters
    ----------
    num_tokens : int
        Maximum sequence length.
    embed_dim : int
        Embedding dimension.

    Returns
    -------
    Tensor
        Positional encoding of shape ``(1, num_tokens, embed_dim)``,
        not registered as a parameter.
    """
    position = torch.arange(num_tokens, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float)
        * (-math.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(1, num_tokens, embed_dim)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


class LearnedPositionalEncoding(nn.Module):
    """Learned (trainable) positional embedding.

    Parameters
    ----------
    num_tokens : int
        Maximum number of token positions.
    embed_dim : int
        Embedding dimension per token.
    """

    def __init__(self, num_tokens: int, embed_dim: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_tokens, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_embed[:, : x.shape[1], :]

    def extra_repr(self) -> str:
        return (
            f"num_tokens={self.pos_embed.shape[1]}, "
            f"embed_dim={self.pos_embed.shape[2]}"
        )
