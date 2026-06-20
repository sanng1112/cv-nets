"""Tests for the LayerScale layer."""

from __future__ import annotations

import torch
from cvnets.layers.layer_scale import LayerScale


class TestLayerScale:
    """Test suite for LayerScale."""

    def test_default_init_near_zero(self) -> None:
        ls = LayerScale(dim=64)
        assert torch.allclose(ls.scale, torch.full((64,), 1e-5))

    def test_custom_init_value(self) -> None:
        ls = LayerScale(dim=32, init_value=0.1)
        assert torch.allclose(ls.scale, torch.full((32,), 0.1))

    def test_forward_scales_channels(self) -> None:
        ls = LayerScale(dim=4, init_value=2.0)
        x = torch.ones(2, 4, 8, 8)
        out = ls(x)
        expected = x * 2.0
        assert torch.allclose(out, expected)

    def test_gradient_flows(self) -> None:
        ls = LayerScale(dim=4, init_value=1.0)
        x = torch.randn(2, 4, 8, 8, requires_grad=False)
        out = ls(x)
        loss = out.sum()
        loss.backward()
        assert ls.scale.grad is not None
        assert not torch.allclose(ls.scale.grad, torch.zeros_like(ls.scale.grad))

    def test_works_with_3d_input(self) -> None:
        ls = LayerScale(dim=16, init_value=1.0)
        x = torch.randn(2, 10, 16)
        out = ls(x)
        assert out.shape == x.shape
