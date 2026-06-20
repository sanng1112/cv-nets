"""Tests for AsymmetricLoss."""
from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import build_loss_fn, SUPPORTED_LOSSES


class TestAsymmetricLoss:
    """Test suite for AsymmetricLoss (multi-label)."""

    def test_registered(self):
        """AsymmetricLoss is registered in SUPPORTED_LOSSES."""
        assert "classification/asymmetric_loss" in SUPPORTED_LOSSES

    def test_build(self):
        """Can build via build_loss_fn."""
        loss_fn = build_loss_fn("asymmetric_loss", category="classification")
        assert loss_fn is not None

    def test_forward_shape_mean(self):
        """Forward returns a scalar (mean reduction)."""
        loss_fn = build_loss_fn("asymmetric_loss", category="classification")
        p = torch.randn(4, 10)
        t = torch.randint(0, 2, (4, 10)).float()
        out = loss_fn(p, t)
        assert out.shape == ()

    def test_forward_sum_reduction(self):
        """Sum reduction works."""
        loss_fn = build_loss_fn("asymmetric_loss", category="classification", reduction="sum")
        p = torch.randn(4, 10)
        t = torch.randint(0, 2, (4, 10)).float()
        out = loss_fn(p, t)
        assert out.shape == ()

    def test_forward_none_reduction(self):
        """None reduction returns per-sample losses."""
        loss_fn = build_loss_fn("asymmetric_loss", category="classification", reduction="none")
        p = torch.randn(4, 10)
        t = torch.randint(0, 2, (4, 10)).float()
        out = loss_fn(p, t)
        assert out.shape == (4,)

    def test_gamma_pos_neg(self):
        """Custom gamma_pos and gamma_neg are accepted."""
        loss_fn = build_loss_fn(
            "asymmetric_loss", category="classification",
            gamma_pos=0.5, gamma_neg=2.0
        )
        p = torch.randn(4, 10)
        t = torch.randint(0, 2, (4, 10)).float()
        out = loss_fn(p, t)
        assert out.shape == ()

    def test_clip_param(self):
        """Clip parameter limits probability range."""
        loss_fn = build_loss_fn(
            "asymmetric_loss", category="classification", clip=0.1
        )
        p = torch.randn(4, 10)
        t = torch.randint(0, 2, (4, 10)).float()
        out = loss_fn(p, t)
        assert out.shape == ()

    def test_loss_non_negative(self):
        """Loss should be non-negative."""
        loss_fn = build_loss_fn("asymmetric_loss", category="classification")
        p = torch.randn(4, 10)
        t = torch.randint(0, 2, (4, 10)).float()
        out = loss_fn(p, t)
        assert out.item() >= 0.0

    def test_all_correct(self):
        """Loss is small when prediction strongly matches target."""
        loss_fn = build_loss_fn("asymmetric_loss", category="classification", reduction="none")
        p = torch.full((4, 10), 10.0)  # high positive logits
        t = torch.ones(4, 10).float()  # all positive labels
        out = loss_fn(p, t)
        # Due to probability clipping (clip=0.05), max prob is 0.95,
        # so minimal loss per class is -log(0.95) ≈ 0.0513
        assert out.mean().item() == pytest.approx(-torch.log(torch.tensor(0.95)).item(), abs=1e-4)

    def test_all_incorrect(self):
        """Loss is > 0 when prediction strongly mismatches target."""
        loss_fn = build_loss_fn("asymmetric_loss", category="classification")
        p = torch.full((4, 10), -10.0)  # strong negative logits
        t = torch.ones(4, 10).float()  # but labels are positive
        out = loss_fn(p, t)
        assert out.item() > 0.0

    def test_extra_repr(self):
        """extra_repr includes key parameters."""
        loss_fn = build_loss_fn(
            "asymmetric_loss", category="classification", gamma_pos=1.0, gamma_neg=3.0
        )
        r = loss_fn.extra_repr()
        assert "gamma_pos" in r
        assert "gamma_neg" in r
