"""Tests for InvertedResidual (MobileNetV2 block)."""

from __future__ import annotations

import torch
from cvnets.blocks.inverted_residual import InvertedResidual


class TestInvertedResidual:
    """Test suite for InvertedResidual."""

    def test_output_shape_same_channels_stride_1(self) -> None:
        """Stride=1, same in/out => skip connection, same spatial size."""
        block = InvertedResidual(
            in_channels=32, out_channels=32, stride=1, expand_ratio=6.0
        )
        x = torch.randn(2, 32, 32, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_output_shape_stride_2(self) -> None:
        """Stride=2 halves spatial, no skip."""
        block = InvertedResidual(
            in_channels=32, out_channels=64, stride=2, expand_ratio=6.0
        )
        x = torch.randn(2, 32, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)

    def test_expansion_channels(self) -> None:
        """Verify the hidden dimension equals round(in_channels * expand_ratio)."""
        in_ch = 24
        exp = 6.0
        hidden = int(round(in_ch * exp))
        block = InvertedResidual(
            in_channels=in_ch, out_channels=in_ch, stride=1, expand_ratio=exp
        )
        # Find the second Conv2d which is the depthwise conv
        conv_count = 0
        depthwise_conv = None
        for m in block.block:
            if isinstance(m, torch.nn.modules.conv.Conv2d):
                conv_count += 1
                if conv_count == 2:
                    depthwise_conv = m
                    break
        assert depthwise_conv is not None, "Could not find depthwise conv"
        assert depthwise_conv.in_channels == hidden, (
            f"Expected depthwise in_channels={hidden}, got "
            f"{depthwise_conv.in_channels}"
        )

    def test_expand_ratio_1_no_expansion(self) -> None:
        """expand_ratio=1: no pointwise expansion."""
        block = InvertedResidual(
            in_channels=64, out_channels=64, stride=1, expand_ratio=1.0
        )
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_skip_connection_condition(self) -> None:
        """Skip active when stride=1 and in==out, off otherwise."""
        b1 = InvertedResidual(
            in_channels=32, out_channels=32, stride=1, expand_ratio=6.0
        )
        assert b1.use_residual is True

        b2 = InvertedResidual(
            in_channels=32, out_channels=64, stride=1, expand_ratio=6.0
        )
        assert b2.use_residual is False

        b3 = InvertedResidual(
            in_channels=32, out_channels=32, stride=2, expand_ratio=6.0
        )
        assert b3.use_residual is False

    def test_hardswish_activation(self) -> None:
        """use_hardswish=True uses Hardswish, otherwise ReLU6."""
        b_hw = InvertedResidual(
            in_channels=16,
            out_channels=16,
            stride=1,
            expand_ratio=6.0,
            use_hardswish=True,
        )
        act_layer = b_hw.block[2]
        assert isinstance(act_layer, torch.nn.Hardswish)

        b_relu = InvertedResidual(
            in_channels=16,
            out_channels=16,
            stride=1,
            expand_ratio=6.0,
            use_hardswish=False,
        )
        act_layer2 = b_relu.block[2]
        assert isinstance(act_layer2, torch.nn.ReLU6)

    def test_projection_has_no_activation(self) -> None:
        """Final 1x1 projection has no activation (last layer is BN)."""
        block = InvertedResidual(
            in_channels=32, out_channels=64, stride=2, expand_ratio=6.0
        )
        last_layer = block.block[-1]
        assert isinstance(last_layer, torch.nn.BatchNorm2d)

    def test_conv_layers_have_gradients(self) -> None:
        """At least one conv parameter gets a non-zero gradient."""
        block = InvertedResidual(
            in_channels=16, out_channels=32, stride=2, expand_ratio=6.0
        )
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        # Check that at least one conv weight has non-zero gradient
        found_nonzero = False
        for name, param in block.named_parameters():
            if "conv" in name.lower() or "weight" in name:
                if param.grad is not None and not torch.allclose(
                    param.grad, torch.zeros_like(param.grad)
                ):
                    found_nonzero = True
                    break
        assert found_nonzero, "No convolution parameters received gradients"
