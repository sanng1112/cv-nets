"""Tests for positional encoding layers."""

from __future__ import annotations

import torch
from cvnets.layers.positional_encoding import (
    sinusoidal_positional_encoding,
    LearnedPositionalEncoding,
)


class TestSinusoidalPositionalEncoding:
    """Test suite for sinusoidal_positional_encoding."""

    def test_output_shape(self) -> None:
        pe = sinusoidal_positional_encoding(num_tokens=196, embed_dim=768)
        assert pe.shape == (1, 196, 768)

    def test_values_in_range(self) -> None:
        pe = sinusoidal_positional_encoding(num_tokens=50, embed_dim=128)
        assert pe.min() >= -1.0
        assert pe.max() <= 1.0

    def test_even_odd_pattern(self) -> None:
        pe = sinusoidal_positional_encoding(num_tokens=10, embed_dim=64)
        assert abs(pe[0, 0, 0].item()) < 1e-6
        assert abs(pe[0, 0, 1].item() - 1.0) < 1e-6

    def test_different_positions_different(self) -> None:
        pe = sinusoidal_positional_encoding(num_tokens=196, embed_dim=768)
        diff = (pe[0, 0, :] - pe[0, 5, :]).abs().sum()
        assert diff > 0.0


class TestLearnedPositionalEncoding:
    """Test suite for LearnedPositionalEncoding."""

    def test_output_shape(self) -> None:
        lpe = LearnedPositionalEncoding(num_tokens=196, embed_dim=768)
        x = torch.randn(2, 196, 768)
        out = lpe(x)
        assert out.shape == (2, 196, 768)

    def test_adds_encoding(self) -> None:
        lpe = LearnedPositionalEncoding(num_tokens=100, embed_dim=64)
        x = torch.ones(2, 100, 64)
        out = lpe(x)
        assert not torch.allclose(out, x)

    def test_different_positions_different_encoding(self) -> None:
        lpe = LearnedPositionalEncoding(num_tokens=10, embed_dim=64)
        diff = (lpe.pos_embed[0, 0, :] - lpe.pos_embed[0, 5, :]).abs().sum()
        assert diff > 0.0

    def test_gradient_flows(self) -> None:
        lpe = LearnedPositionalEncoding(num_tokens=50, embed_dim=128)
        x = torch.randn(2, 50, 128)
        out = lpe(x)
        loss = out.sum()
        loss.backward()
        assert lpe.pos_embed.grad is not None
        assert not torch.allclose(
            lpe.pos_embed.grad, torch.zeros_like(lpe.pos_embed.grad)
        )
