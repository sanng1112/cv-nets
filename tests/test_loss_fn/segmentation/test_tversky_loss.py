"""Tests for TverskyLoss."""

from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import build_loss_fn, SUPPORTED_LOSSES


class TestTverskyLoss:
    """Test suite for TverskyLoss."""

    def test_registered(self):
        """TverskyLoss is registered in SUPPORTED_LOSSES."""
        assert "segmentation/tversky_loss" in SUPPORTED_LOSSES

    def test_build(self):
        """Can build via build_loss_fn."""
        fn = build_loss_fn("tversky_loss", category="segmentation")
        assert fn is not None

    def test_basic(self):
        """Forward returns a scalar in [0, 1] range."""
        fn = build_loss_fn("tversky_loss", category="segmentation")
        out = fn(torch.randn(4, 5, 32, 32), torch.randint(0, 5, (4, 32, 32)))
        assert out.shape == ()
        assert 0.0 <= out.item() <= 1.0

    def test_equals_dice_when_05(self):
        """With alpha=beta=0.5, Tversky equals Dice."""
        tv = build_loss_fn(
            "tversky_loss", category="segmentation", alpha=0.5, beta=0.5
        )
        dc = build_loss_fn("dice_loss", category="segmentation")
        torch.manual_seed(42)
        p = torch.randn(2, 3, 8, 8)
        t = torch.randint(0, 3, (2, 8, 8))
        assert torch.allclose(tv(p, t), dc(p, t), atol=1e-4)

    def test_asymmetric_alpha(self):
        """Different alpha/beta produces different loss."""
        fn1 = build_loss_fn(
            "tversky_loss", category="segmentation", alpha=0.3, beta=0.7
        )
        fn2 = build_loss_fn(
            "tversky_loss", category="segmentation", alpha=0.7, beta=0.3
        )
        torch.manual_seed(42)
        p = torch.randn(2, 3, 8, 8)
        t = torch.randint(0, 3, (2, 8, 8))
        # Should give different results with swapped alpha/beta
        assert not torch.allclose(fn1(p, t), fn2(p, t), atol=1e-4)

    def test_forward_sum_reduction(self):
        """Sum reduction works."""
        fn = build_loss_fn("tversky_loss", category="segmentation", reduction="sum")
        out = fn(torch.randn(4, 5, 32, 32), torch.randint(0, 5, (4, 32, 32)))
        assert out.shape == ()

    def test_forward_none_reduction(self):
        """None reduction returns per-batch losses."""
        fn = build_loss_fn(
            "tversky_loss", category="segmentation", reduction="none"
        )
        out = fn(torch.randn(4, 5, 32, 32), torch.randint(0, 5, (4, 32, 32)))
        assert out.shape == (4,)

    def test_extra_repr(self):
        """extra_repr includes key parameters."""
        fn = build_loss_fn(
            "tversky_loss", category="segmentation", alpha=0.3, beta=0.7
        )
        r = fn.extra_repr()
        assert "alpha" in r
        assert "beta" in r
