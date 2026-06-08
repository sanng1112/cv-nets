"""Tests for TripletLoss."""
from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import build_loss_fn, SUPPORTED_LOSSES


class TestTripletLoss:
    """Test suite for TripletLoss."""

    def test_registered(self):
        """TripletLoss is registered in SUPPORTED_LOSSES."""
        assert "metric_learning/triplet_loss" in SUPPORTED_LOSSES

    def test_build(self):
        """Can build via build_loss_fn."""
        loss_fn = build_loss_fn("triplet_loss", category="metric_learning")
        assert loss_fn is not None

    def test_basic(self):
        """Forward returns a non-negative scalar (mean reduction)."""
        fn = build_loss_fn("triplet_loss", category="metric_learning", margin=1.)
        out = fn(torch.randn(12, 32), torch.randint(0, 3, (12,)))
        assert out.shape == ()
        assert out.item() >= 0

    def test_separated_zero(self):
        """Well-separated embeddings (one-hot per class) produce near-zero loss."""
        fn = build_loss_fn("triplet_loss", category="metric_learning", margin=1.)
        # Each class occupies a different axis => dot products are zero
        emb = torch.zeros(6, 3)
        emb[0, 0] = 1.0  # class 0: axis 0
        emb[1, 0] = 1.0
        emb[2, 1] = 1.0  # class 1: axis 1
        emb[3, 1] = 1.0
        emb[4, 2] = 1.0  # class 2: axis 2
        emb[5, 2] = 1.0
        out = fn(emb, torch.tensor([0, 0, 1, 1, 2, 2]))
        assert out.item() < 0.1

    def test_semi_hard(self):
        """Semi-hard mining strategy works."""
        fn = build_loss_fn("triplet_loss", category="metric_learning", margin=0.5, mining="semi_hard")
        assert fn(torch.randn(12, 16), torch.randint(0, 3, (12,))).shape == ()

    def test_all_strategy(self):
        """All valid triplets averaging works."""
        fn = build_loss_fn("triplet_loss", category="metric_learning", margin=0.5, mining="all")
        assert fn(torch.randn(12, 16), torch.randint(0, 3, (12,))).shape == ()

    def test_gradient(self):
        """Gradient flows through embeddings."""
        fn = build_loss_fn("triplet_loss", category="metric_learning", margin=1.)
        emb = torch.randn(12, 16, requires_grad=True)
        fn(emb, torch.randint(0, 3, (12,))).backward()
        assert emb.grad is not None

    def test_forward_sum_reduction(self):
        """Sum reduction works."""
        fn = build_loss_fn("triplet_loss", category="metric_learning", reduction="sum")
        out = fn(torch.randn(12, 32), torch.randint(0, 3, (12,)))
        assert out.shape == ()

    def test_forward_none_reduction(self):
        """None reduction returns per-sample losses."""
        fn = build_loss_fn("triplet_loss", category="metric_learning", reduction="none")
        out = fn(torch.randn(12, 32), torch.randint(0, 3, (12,)))
        assert out.shape == (12,)

    def test_extra_repr(self):
        """extra_repr includes key parameters."""
        fn = build_loss_fn("triplet_loss", category="metric_learning", margin=1.5, mining="hard")
        r = fn.extra_repr()
        assert "margin" in r and "mining" in r
