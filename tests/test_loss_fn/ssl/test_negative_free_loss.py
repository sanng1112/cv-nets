"""Tests for NegativeFreeLoss (BYOL / SimSiam)."""
from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import build_loss_fn, SUPPORTED_LOSSES


class TestNegativeFreeLoss:
    """Test suite for NegativeFreeLoss."""

    def test_registered(self):
        """NegativeFreeLoss is registered in SUPPORTED_LOSSES."""
        assert "ssl/negative_free_loss" in SUPPORTED_LOSSES

    def test_build_default(self):
        """Can build via build_loss_fn with default mode."""
        fn = build_loss_fn("negative_free_loss", category="ssl")
        assert fn is not None
        assert fn.mode == "byol"

    def test_build_simsiam(self):
        """Can build with mode='simsiam'."""
        fn = build_loss_fn("negative_free_loss", category="ssl", mode="simsiam")
        assert fn.mode == "simsiam"

    def test_basic_byol(self):
        """Forward returns a scalar (mean reduction) with BYOL mode."""
        fn = build_loss_fn("negative_free_loss", category="ssl", mode="byol")
        out = fn(torch.randn(4, 64), torch.randn(4, 64))
        assert out.shape == ()

    def test_basic_simsiam(self):
        """Forward returns a scalar (mean reduction) with SimSiam mode."""
        fn = build_loss_fn("negative_free_loss", category="ssl", mode="simsiam")
        out = fn(torch.randn(4, 64), torch.randn(4, 64))
        assert out.shape == ()

    def test_identical_embeddings_low_loss(self):
        """Identical embeddings produce near-zero loss."""
        fn = build_loss_fn("negative_free_loss", category="ssl")
        e = torch.randn(4, 32)
        out = fn(e, e)
        assert out.item() < 0.05

    def test_opposite_embeddings_high_loss(self):
        """Opposite embeddings produce high loss (near 4)."""
        fn = build_loss_fn("negative_free_loss", category="ssl")
        # prediction and target are negatives of each other -> cos = -1
        p = torch.randn(4, 32)
        z = -p
        out = fn(p, z)
        # 2 - 2*(-1) = 4
        assert torch.allclose(out, torch.tensor(4.0), atol=1e-5)

    def test_byol_gradient(self):
        """In BYOL mode, gradient flows through prediction but not target."""
        fn = build_loss_fn("negative_free_loss", category="ssl", mode="byol")
        p = torch.randn(4, 32, requires_grad=True)
        z = torch.randn(4, 32)
        out = fn(p, z)
        out.backward()
        assert p.grad is not None
        # grad on target should be None (no gradient flow)
        assert z.grad is None

    def test_simsiam_gradient_both(self):
        """In SimSiam mode, gradient flows through both prediction and target."""
        fn = build_loss_fn("negative_free_loss", category="ssl", mode="simsiam")
        p = torch.randn(4, 32, requires_grad=True)
        z = torch.randn(4, 32, requires_grad=True)
        out = fn(p, z)
        out.backward()
        assert p.grad is not None
        assert z.grad is not None

    def test_forward_sum_reduction(self):
        """Sum reduction works."""
        fn = build_loss_fn("negative_free_loss", category="ssl", reduction="sum")
        out = fn(torch.randn(4, 64), torch.randn(4, 64))
        assert out.shape == ()

    def test_forward_none_reduction(self):
        """None reduction returns per-sample losses."""
        fn = build_loss_fn("negative_free_loss", category="ssl", reduction="none")
        out = fn(torch.randn(4, 64), torch.randn(4, 64))
        assert out.shape == (4,)

    def test_extra_repr(self):
        """extra_repr includes mode and reduction."""
        fn = build_loss_fn("negative_free_loss", category="ssl", mode="simsiam")
        r = fn.extra_repr()
        assert "mode" in r and "simsiam" in r
        assert "reduction" in r

    def test_invalid_mode_raises(self):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError):
            build_loss_fn("negative_free_loss", category="ssl", mode="invalid")
