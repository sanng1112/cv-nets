"""Tests for ComboLoss."""

from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import build_loss_fn, SUPPORTED_LOSSES


class TestComboLoss:
    """Test suite for ComboLoss."""

    def test_registered(self):
        """ComboLoss is registered in SUPPORTED_LOSSES."""
        assert "segmentation/combo_loss" in SUPPORTED_LOSSES

    def test_build(self):
        """Can build via build_loss_fn."""
        fn = build_loss_fn("combo_loss", category="segmentation")
        assert fn is not None

    def test_basic(self):
        """Forward returns a scalar."""
        fn = build_loss_fn("combo_loss", category="segmentation")
        out = fn(torch.randn(4, 5, 32, 32), torch.randint(0, 5, (4, 32, 32)))
        assert out.shape == ()

    def test_default_weight(self):
        """With alpha=0.5, ComboLoss equals 0.5*CE + 0.5*Dice."""
        fn = build_loss_fn("combo_loss", category="segmentation")
        dc = build_loss_fn("dice_loss", category="segmentation")
        ce = build_loss_fn("cross_entropy", category="classification")
        torch.manual_seed(42)
        p = torch.randn(2, 3, 16, 16)
        t = torch.randint(0, 3, (2, 16, 16))
        expected = 0.5 * ce(p, t) + 0.5 * dc(p, t)
        assert torch.allclose(fn(p, t), expected, atol=1e-4)

    def test_alpha_zero(self):
        """alpha=0 gives pure Dice loss."""
        fn = build_loss_fn("combo_loss", category="segmentation", alpha=0.0)
        dc = build_loss_fn("dice_loss", category="segmentation")
        torch.manual_seed(42)
        p = torch.randn(2, 3, 16, 16)
        t = torch.randint(0, 3, (2, 16, 16))
        assert torch.allclose(fn(p, t), dc(p, t), atol=1e-4)

    def test_alpha_one(self):
        """alpha=1 gives pure CE loss."""
        fn = build_loss_fn("combo_loss", category="segmentation", alpha=1.0)
        ce = build_loss_fn("cross_entropy", category="classification")
        torch.manual_seed(42)
        p = torch.randn(2, 3, 16, 16)
        t = torch.randint(0, 3, (2, 16, 16))
        assert torch.allclose(fn(p, t), ce(p, t), atol=1e-4)

    def test_forward_sum_reduction(self):
        """Sum reduction works."""
        fn = build_loss_fn("combo_loss", category="segmentation", reduction="sum")
        out = fn(torch.randn(4, 5, 32, 32), torch.randint(0, 5, (4, 32, 32)))
        assert out.shape == ()

    def test_forward_none_reduction(self):
        """None reduction returns per-batch losses."""
        fn = build_loss_fn("combo_loss", category="segmentation", reduction="none")
        out = fn(torch.randn(4, 5, 32, 32), torch.randint(0, 5, (4, 32, 32)))
        assert out.shape == (4,)

    def test_extra_repr(self):
        """extra_repr includes key parameters."""
        fn = build_loss_fn(
            "combo_loss", category="segmentation", alpha=0.3, smooth=1e-5
        )
        r = fn.extra_repr()
        assert "alpha" in r
        assert "smooth" in r
