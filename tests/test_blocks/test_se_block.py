"""Tests for the Squeeze-and-Excitation block."""

from __future__ import annotations

import torch
from cvnets.blocks.se_block import SEBlock


class TestSEBlock:
    """Test suite for SEBlock."""

    def test_output_shape_matches_input(self) -> None:
        se = SEBlock(in_channels=64, reduction=16)
        x = torch.randn(2, 64, 32, 32)
        out = se(x)
        assert out.shape == x.shape

    def test_excitation_in_01(self) -> None:
        se = SEBlock(in_channels=64, reduction=16)
        se.eval()
        x = torch.randn(4, 64, 16, 16)
        out = se(x)
        assert torch.all(out.abs() <= x.abs() + 1e-6)

    def test_reduction_ratio_shrinks_params(self) -> None:
        se_small = SEBlock(in_channels=64, reduction=4)
        se_large = SEBlock(in_channels=64, reduction=16)
        params_small = sum(p.numel() for p in se_small.parameters())
        params_large = sum(p.numel() for p in se_large.parameters())
        assert params_small > params_large

    def test_reduction_one_means_no_bottleneck(self) -> None:
        se = SEBlock(in_channels=32, reduction=1)
        x = torch.randn(2, 32, 8, 8)
        out = se(x)
        assert out.shape == x.shape

    def test_gradient_flows(self) -> None:
        se = SEBlock(in_channels=32, reduction=8)
        x = torch.randn(2, 32, 16, 16)
        out = se(x)
        loss = out.sum()
        loss.backward()
        for name, param in se.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.allclose(
                param.grad, torch.zeros_like(param.grad)
            ), f"{name} gradient is zero"
