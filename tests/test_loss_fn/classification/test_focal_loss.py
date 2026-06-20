"""Tests for FocalLoss."""
from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import build_loss_fn, SUPPORTED_LOSSES


class TestFocalLoss:
    """Test suite for FocalLoss."""

    def test_registered(self):
        """FocalLoss is registered in SUPPORTED_LOSSES."""
        assert "classification/focal_loss" in SUPPORTED_LOSSES

    def test_build(self):
        """Can build via build_loss_fn."""
        loss_fn = build_loss_fn("focal_loss", category="classification")
        assert loss_fn is not None

    def test_forward_shape(self):
        """Forward returns a scalar (mean reduction)."""
        loss_fn = build_loss_fn("focal_loss", category="classification")
        p = torch.randn(4, 10)
        t = torch.randint(0, 10, (4,))
        out = loss_fn(p, t)
        assert out.shape == ()

    def test_forward_sum_reduction(self):
        """Sum reduction works."""
        loss_fn = build_loss_fn("focal_loss", category="classification", reduction="sum")
        p = torch.randn(4, 10)
        t = torch.randint(0, 10, (4,))
        out = loss_fn(p, t)
        assert out.shape == ()

    def test_forward_none_reduction(self):
        """None reduction returns per-sample losses."""
        loss_fn = build_loss_fn("focal_loss", category="classification", reduction="none")
        p = torch.randn(4, 10)
        t = torch.randint(0, 10, (4,))
        out = loss_fn(p, t)
        assert out.shape == (4,)

    def test_gamma_default(self):
        """Default gamma=2.0 produces valid loss."""
        loss_fn = build_loss_fn("focal_loss", category="classification")
        p = torch.randn(4, 10)
        t = torch.randint(0, 10, (4,))
        out = loss_fn(p, t)
        assert out.shape == ()
        assert out.item() >= 0.0

    def test_gamma_zero(self):
        """gamma=0 makes focal loss equivalent to cross-entropy."""
        focal_fn = build_loss_fn("focal_loss", category="classification", gamma=0.0)
        ce_fn = build_loss_fn("cross_entropy", category="classification")
        p = torch.randn(4, 10)
        t = torch.randint(0, 10, (4,))
        focal_out = focal_fn(p.clone(), t.clone())
        ce_out = ce_fn(p.clone(), t.clone())
        assert focal_out.item() == pytest.approx(ce_out.item(), rel=1e-5)

    def test_alpha_scalar(self):
        """Scalar alpha is accepted."""
        loss_fn = build_loss_fn("focal_loss", category="classification", alpha=0.25)
        p = torch.randn(4, 10)
        t = torch.randint(0, 10, (4,))
        out = loss_fn(p, t)
        assert out.shape == ()

    def test_alpha_list(self):
        """List alpha is accepted."""
        alpha = [0.5] * 10
        loss_fn = build_loss_fn("focal_loss", category="classification", alpha=alpha)
        p = torch.randn(4, 10)
        t = torch.randint(0, 10, (4,))
        out = loss_fn(p, t)
        assert out.shape == ()

    def test_ignore_index(self):
        """ignore_index is handled."""
        loss_fn = build_loss_fn(
            "focal_loss", category="classification", ignore_index=-1, reduction="none"
        )
        p = torch.randn(5, 10)
        t = torch.tensor([0, 1, -1, 3, 4])
        out = loss_fn(p, t)
        assert out.shape == (5,)
        assert out[2].item() == pytest.approx(0.0)

    def test_extra_repr(self):
        """extra_repr includes key parameters."""
        loss_fn = build_loss_fn(
            "focal_loss", category="classification", gamma=3.0, alpha=0.5
        )
        r = loss_fn.extra_repr()
        assert "gamma" in r
