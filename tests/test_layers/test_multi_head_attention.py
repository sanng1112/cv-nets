"""Tests for the MultiHeadSelfAttention layer."""

from __future__ import annotations

import torch
from cvnets.layers.multi_head_attention import MultiHeadSelfAttention


class TestMultiHeadSelfAttention:
    """Test suite for MultiHeadSelfAttention."""

    def test_output_shape(self) -> None:
        mhsa = MultiHeadSelfAttention(embed_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        out = mhsa(x)
        assert out.shape == (2, 10, 64)

    def test_qkv_projection_exists(self) -> None:
        mhsa = MultiHeadSelfAttention(embed_dim=64, num_heads=4)
        assert hasattr(mhsa, "qkv")
        assert mhsa.qkv.weight.shape == (64 * 3, 64)

    def test_output_projection_exists(self) -> None:
        mhsa = MultiHeadSelfAttention(embed_dim=64, num_heads=4)
        assert mhsa.proj.weight.shape == (64, 64)

    def test_gradient_flows(self) -> None:
        mhsa = MultiHeadSelfAttention(embed_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        out = mhsa(x)
        loss = out.sum()
        loss.backward()
        for name, param in mhsa.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), \
                f"{name} gradient is zero"

    def test_causal_mask(self) -> None:
        mhsa = MultiHeadSelfAttention(embed_dim=32, num_heads=2)
        x = torch.randn(1, 5, 32)
        x1 = x.clone()
        out1 = mhsa(x1, causal_mask=True)
        x2 = x.clone()
        x2[0, 4, :] = 999.0
        out2 = mhsa(x2, causal_mask=True)
        assert torch.allclose(out1[0, :4, :], out2[0, :4, :], atol=1e-4)

    def test_no_causal_mask_allows_full_attention(self) -> None:
        mhsa = MultiHeadSelfAttention(embed_dim=32, num_heads=2)
        x1 = torch.randn(1, 5, 32)
        out1 = mhsa(x1, causal_mask=False)
        x2 = x1.clone()
        x2[0, 4, :] = 999.0
        out2 = mhsa(x2, causal_mask=False)
        diff = (out1 - out2).abs().max().item()
        assert diff > 0.01, "Expected earlier positions to change with full attention"

    def test_different_embed_dim_and_heads(self) -> None:
        mhsa = MultiHeadSelfAttention(embed_dim=256, num_heads=8)
        x = torch.randn(4, 20, 256)
        out = mhsa(x)
        assert out.shape == (4, 20, 256)
