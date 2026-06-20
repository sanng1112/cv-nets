"""Tests for BottleneckBlock."""

from __future__ import annotations

import torch
from cvnets.blocks.bottleneck_block import BottleneckBlock


class TestBottleneckBlock:
    """Test suite for BottleneckBlock."""

    def test_output_shape_basic(self) -> None:
        """Default expansion=4: output channels = in * 4."""
        block = BottleneckBlock(in_channels=64, out_channels=64, expansion=4, stride=1)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 256, 32, 32)

    def test_channel_progression(self) -> None:
        """Verify the three conv stages produce correct intermediate shapes."""
        in_ch = 32
        out_ch = 64
        exp = 4
        block = BottleneckBlock(in_channels=in_ch, out_channels=out_ch, expansion=exp, stride=1)
        x = torch.randn(2, in_ch, 16, 16)

        # 1x1 reduce: out_ch
        h = block.conv1(x)
        assert h.shape[1] == out_ch, f"conv1 expected {out_ch} channels, got {h.shape[1]}"

        # 3x3: out_ch
        h = block.conv2(block.norm1(h))
        assert h.shape[1] == out_ch, f"conv2 expected {out_ch} channels, got {h.shape[1]}"

        # 1x1 expand: out_ch * exp
        h = block.conv3(block.norm2(h))
        assert h.shape[1] == out_ch * exp, f"conv3 expected {out_ch * exp} channels, got {h.shape[1]}"

    def test_stride_2(self) -> None:
        """Stride=2 halves spatial dimension."""
        block = BottleneckBlock(in_channels=64, out_channels=128, expansion=4, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 512, 16, 16)

    def test_downsample_when_needed(self) -> None:
        """Downsample is active when stride != 1 or in_channels != expanded_dim."""
        block1 = BottleneckBlock(in_channels=32, out_channels=64, expansion=4, stride=1)
        assert not isinstance(block1.downsample, torch.nn.Identity)

        block2 = BottleneckBlock(in_channels=256, out_channels=64, expansion=4, stride=1)
        assert isinstance(block2.downsample, torch.nn.Identity)

    def test_expansion_1(self) -> None:
        """Expansion=1: output channels = out_channels."""
        block = BottleneckBlock(in_channels=64, out_channels=128, expansion=1, stride=1)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == (2, 128, 16, 16)

    def test_gradient_flows(self) -> None:
        """Backward pass produces non-zero gradients."""
        block = BottleneckBlock(in_channels=16, out_channels=32, expansion=2, stride=1)
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"{name} gradient is zero"
