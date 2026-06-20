"""Tests for ResidualBlock (BasicBlock)."""

from __future__ import annotations

import torch
from cvnets.blocks.residual_block import ResidualBlock


class TestResidualBlock:
    """Test suite for ResidualBlock."""

    def test_output_shape_same_channels(self) -> None:
        """Same in/out channels, stride=1 => output shape matches input."""
        block = ResidualBlock(in_channels=64, out_channels=64, stride=1)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_output_shape_different_channels(self) -> None:
        """Different in/out channels triggers downsample."""
        block = ResidualBlock(in_channels=32, out_channels=64, stride=1)
        x = torch.randn(2, 32, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)

    def test_stride_2_downsample(self) -> None:
        """Stride=2 halves spatial dimension."""
        block = ResidualBlock(in_channels=64, out_channels=128, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 16, 16)

    def test_skip_connection_active(self) -> None:
        """Verify that skip connection (downsample) changes channels."""
        block = ResidualBlock(in_channels=32, out_channels=64, stride=1)
        assert not isinstance(block.downsample, torch.nn.Identity)

    def test_skip_connection_identity(self) -> None:
        """When in==out and stride==1, downsample is Identity."""
        block = ResidualBlock(in_channels=64, out_channels=64, stride=1)
        assert isinstance(block.downsample, torch.nn.Identity)

    def test_no_norm(self) -> None:
        """Block works without normalisation layers."""
        block = ResidualBlock(
            in_channels=16, out_channels=32, stride=1, use_norm=False
        )
        x = torch.randn(2, 16, 16, 16)
        out = block(x)
        assert out.shape == (2, 32, 16, 16)

    def test_gradient_flows(self) -> None:
        """Backward pass produces non-zero gradients."""
        block = ResidualBlock(in_channels=16, out_channels=32, stride=2)
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"{name} gradient is zero"
