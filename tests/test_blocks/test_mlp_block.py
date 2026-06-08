"""Tests for the Transformer MLP block."""

from __future__ import annotations

import torch
from cvnets.blocks.mlp_block import TransformerMLP


class TestTransformerMLP:
    """Test suite for TransformerMLP."""

    def test_output_shape_matches_input(self) -> None:
        mlp = TransformerMLP(embed_dim=128, expansion_ratio=4)
        x = torch.randn(2, 10, 128)
        out = mlp(x)
        assert out.shape == (2, 10, 128)

    def test_expansion_increases_hidden_dim(self) -> None:
        mlp = TransformerMLP(embed_dim=64, expansion_ratio=4)
        assert mlp.fc1.out_features == 256

    def test_gaussian_error_linear_unit_used(self) -> None:
        mlp = TransformerMLP(embed_dim=64, expansion_ratio=4)
        from torch.nn import GELU

        assert isinstance(mlp.act, GELU)

    def test_dropout_is_applied(self) -> None:
        mlp = TransformerMLP(embed_dim=64, expansion_ratio=4, dropout=0.5)
        mlp.train()
        torch.manual_seed(42)
        x = torch.ones(100, 5, 64)
        out1 = mlp(x)
        out2 = mlp(x)
        assert not torch.allclose(out1, out2)

    def test_gradient_flows(self) -> None:
        mlp = TransformerMLP(embed_dim=32, expansion_ratio=2)
        x = torch.randn(2, 10, 32)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        for name, param in mlp.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.allclose(
                param.grad, torch.zeros_like(param.grad)
            ), f"{name} gradient is zero"
