"""Tests for the RMSNorm normalisation layer."""

from __future__ import annotations

import torch
from cvnets.layers.normalization import (
    build_normalization_layer,
    SUPPORTED_NORM_FNS,
)


class TestRMSNorm:
    """Test suite for RMSNorm."""

    def test_registered_in_supported(self) -> None:
        assert "rms_norm" in SUPPORTED_NORM_FNS

    def test_build_via_factory(self) -> None:
        layer = build_normalization_layer(
            opts={"type": "rms_norm"},
            num_features=64,
        )
        assert layer is not None
        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert out.shape == (2, 10, 64)

    def test_normalizes_to_unit_variance_approx(self) -> None:
        from cvnets.layers.normalization.rms_norm import RMSNorm

        rms = RMSNorm(normalized_shape=64)
        rms.eval()
        x = torch.randn(4, 10, 64) * 5.0 + 3.0
        out = rms(x)
        rms_values = torch.sqrt(torch.mean(out ** 2, dim=-1))
        assert torch.allclose(rms_values, torch.ones_like(rms_values), atol=0.1)

    def test_gradient_flows(self) -> None:
        from cvnets.layers.normalization.rms_norm import RMSNorm

        rms = RMSNorm(normalized_shape=64)
        x = torch.randn(2, 10, 64)
        out = rms(x)
        loss = out.sum()
        loss.backward()
        assert rms.weight.grad is not None
        assert not torch.allclose(
            rms.weight.grad, torch.zeros_like(rms.weight.grad)
        )

    def test_with_eps(self) -> None:
        from cvnets.layers.normalization.rms_norm import RMSNorm

        rms = RMSNorm(normalized_shape=64, eps=1e-3)
        x = torch.randn(2, 10, 64)
        out = rms(x)
        assert out.shape == (2, 10, 64)
