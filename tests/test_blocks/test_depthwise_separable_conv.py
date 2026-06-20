"""Tests for DepthwiseSeparableConvBlock."""

from __future__ import annotations

import torch
from cvnets.blocks.depthwise_separable_conv import DepthwiseSeparableConvBlock


class TestDepthwiseSeparableConvBlock:
    """Test suite for DepthwiseSeparableConvBlock."""

    def test_output_shape_basic(self) -> None:
        block = DepthwiseSeparableConvBlock(
            in_channels=16, out_channels=32, kernel_size=3, stride=1
        )
        x = torch.randn(2, 16, 32, 32)
        out = block(x)
        assert out.shape == (2, 32, 32, 32)

    def test_output_shape_stride_2(self) -> None:
        block = DepthwiseSeparableConvBlock(
            in_channels=8, out_channels=16, kernel_size=3, stride=2
        )
        x = torch.randn(2, 8, 64, 64)
        out = block(x)
        assert out.shape == (2, 16, 32, 32)

    def test_fewer_parameters_than_regular_conv(self) -> None:
        dw_block = DepthwiseSeparableConvBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
        )
        dw_params = sum(p.numel() for p in dw_block.parameters())
        regular_conv_params = 64 * 64 * 3 * 3
        assert dw_params < regular_conv_params

    def test_gradient_flows(self) -> None:
        block = DepthwiseSeparableConvBlock(
            in_channels=16, out_channels=32, kernel_size=3, stride=1
        )
        x = torch.randn(2, 16, 8, 8)
        out = block(x)
        loss = out.sum()
        loss.backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.allclose(
                param.grad, torch.zeros_like(param.grad)
            ), f"{name} gradient is zero"
