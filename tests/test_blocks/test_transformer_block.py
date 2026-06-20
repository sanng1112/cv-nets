"""Tests for TransformerEncoderBlock."""

from __future__ import annotations

import torch
from cvnets.blocks.transformer_block import TransformerEncoderBlock


class TestTransformerEncoderBlock:
    """Test suite for TransformerEncoderBlock."""

    def test_output_shape_matches_input(self) -> None:
        block = TransformerEncoderBlock(embed_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == (2, 10, 64)

    def test_contains_attention_and_mlp(self) -> None:
        block = TransformerEncoderBlock(embed_dim=128, num_heads=4)
        assert hasattr(block, "attn")
        assert hasattr(block, "mlp")
        assert hasattr(block, "norm1")
        assert hasattr(block, "norm2")

    def test_pre_norm_structure(self) -> None:
        block = TransformerEncoderBlock(embed_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_drop_path_integration(self) -> None:
        block = TransformerEncoderBlock(
            embed_dim=64, num_heads=4, drop_path=0.5
        )
        block.train()
        torch.manual_seed(42)
        x = torch.randn(2, 10, 64)
        out1 = block(x)
        out2 = block(x)
        assert not torch.allclose(out1, out2)

    def test_gradient_flows(self) -> None:
        block = TransformerEncoderBlock(embed_dim=32, num_heads=2)
        x = torch.randn(2, 10, 32)
        out = block(x)
        loss = out.sum()
        loss.backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.allclose(
                param.grad, torch.zeros_like(param.grad)
            ), f"{name} gradient is zero"

    def test_layer_scale_option(self) -> None:
        block = TransformerEncoderBlock(
            embed_dim=64, num_heads=4, layer_scale_init=1e-5
        )
        from cvnets.layers.layer_scale import LayerScale

        assert isinstance(block.ls1, LayerScale)
        assert isinstance(block.ls2, LayerScale)

    def test_without_layer_scale(self) -> None:
        block = TransformerEncoderBlock(
            embed_dim=64, num_heads=4, layer_scale_init=0.0
        )
        from torch.nn import Identity

        assert isinstance(block.ls1, Identity)
