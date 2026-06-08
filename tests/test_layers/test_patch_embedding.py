"""Tests for the PatchEmbedding layer."""

from __future__ import annotations

import torch
from cvnets.layers.patch_embedding import PatchEmbedding


class TestPatchEmbedding:
    """Test suite for PatchEmbedding."""

    def test_output_shape(self) -> None:
        pe = PatchEmbedding(
            img_size=224, patch_size=16, in_channels=3, embed_dim=768
        )
        x = torch.randn(2, 3, 224, 224)
        out = pe(x)
        assert out.shape == (2, 196, 768)

    def test_square_patches(self) -> None:
        pe = PatchEmbedding(
            img_size=32, patch_size=8, in_channels=1, embed_dim=128
        )
        x = torch.randn(4, 1, 32, 32)
        out = pe(x)
        assert out.shape == (4, 16, 128)

    def test_rectangular_image(self) -> None:
        pe = PatchEmbedding(
            img_size=(64, 128), patch_size=16, in_channels=3, embed_dim=256
        )
        x = torch.randn(2, 3, 64, 128)
        out = pe(x)
        assert out.shape == (2, 32, 256)

    def test_gradient_flows(self) -> None:
        pe = PatchEmbedding(
            img_size=32, patch_size=8, in_channels=3, embed_dim=128
        )
        x = torch.randn(2, 3, 32, 32)
        out = pe(x)
        loss = out.sum()
        loss.backward()
        assert pe.proj.weight.grad is not None
        assert not torch.allclose(
            pe.proj.weight.grad, torch.zeros_like(pe.proj.weight.grad)
        )

    def test_num_patches_property(self) -> None:
        pe = PatchEmbedding(
            img_size=224, patch_size=16, in_channels=3, embed_dim=768
        )
        assert pe.num_patches == 196
