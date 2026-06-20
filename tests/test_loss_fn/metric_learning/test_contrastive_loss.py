"""Tests for ContrastiveLoss."""
from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import build_loss_fn, SUPPORTED_LOSSES


class TestContrastiveLoss:
    """Test suite for ContrastiveLoss."""

    def test_registered(self):
        """ContrastiveLoss is registered in SUPPORTED_LOSSES."""
        assert "metric_learning/contrastive_loss" in SUPPORTED_LOSSES

    def test_build(self):
        """Can build via build_loss_fn."""
        fn = build_loss_fn("contrastive_loss", category="metric_learning")
        assert fn is not None

    def test_basic(self):
        """Forward returns a positive scalar (mean reduction)."""
        fn = build_loss_fn("contrastive_loss", category="metric_learning")
        out = fn(torch.randn(4, 32), torch.randn(4, 32), label=torch.randint(0, 2, (4,)).float())
        assert out.shape == ()
        assert out.item() > 0

    def test_same_low(self):
        """Identical embeddings with positive label yield near-zero loss."""
        fn = build_loss_fn("contrastive_loss", category="metric_learning", margin=2.)
        emb = torch.randn(4, 16)
        out = fn(emb, emb, label=torch.ones(4))
        assert out.item() < 0.05

    def test_different_high(self):
        """Dissimilar embeddings with positive label yield high loss."""
        fn = build_loss_fn("contrastive_loss", category="metric_learning", margin=2.)
        e1 = torch.randn(4, 16)
        e2 = -e1  # opposite direction
        out = fn(e1, e2, label=torch.ones(4))
        # distance should be large => loss close to margin^2/2 = 2
        assert out.item() > 1.0

    def test_negative_zero(self):
        """Negative pair with distance > margin yields zero loss."""
        fn = build_loss_fn("contrastive_loss", category="metric_learning", margin=2.)
        e1 = torch.randn(4, 16)
        e2 = -e1  # far apart
        out = fn(e1, e2, label=torch.zeros(4))
        assert out.item() < 0.05

    def test_gradient(self):
        """Gradient flows through both embeddings."""
        fn = build_loss_fn("contrastive_loss", category="metric_learning")
        e1 = torch.randn(4, 16, requires_grad=True)
        e2 = torch.randn(4, 16, requires_grad=True)
        fn(e1, e2, label=torch.randint(0, 2, (4,)).float()).backward()
        assert e1.grad is not None
        assert e2.grad is not None

    def test_forward_sum_reduction(self):
        """Sum reduction works."""
        fn = build_loss_fn("contrastive_loss", category="metric_learning", reduction="sum")
        out = fn(torch.randn(4, 32), torch.randn(4, 32), label=torch.randint(0, 2, (4,)).float())
        assert out.shape == ()

    def test_forward_none_reduction(self):
        """None reduction returns per-sample losses."""
        fn = build_loss_fn("contrastive_loss", category="metric_learning", reduction="none")
        out = fn(torch.randn(4, 32), torch.randn(4, 32), label=torch.randint(0, 2, (4,)).float())
        assert out.shape == (4,)

    def test_extra_repr(self):
        """extra_repr includes key parameters."""
        fn = build_loss_fn("contrastive_loss", category="metric_learning", margin=1.5)
        r = fn.extra_repr()
        assert "margin" in r
