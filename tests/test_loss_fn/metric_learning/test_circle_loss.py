"""Tests for CircleLoss."""
from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import build_loss_fn, SUPPORTED_LOSSES


class TestCircleLoss:
    """Test suite for CircleLoss."""

    def test_registered(self):
        """CircleLoss is registered in SUPPORTED_LOSSES."""
        assert "metric_learning/circle_loss" in SUPPORTED_LOSSES

    def test_build(self):
        """Can build via build_loss_fn."""
        fn = build_loss_fn("circle_loss", category="metric_learning")
        assert fn is not None

    def test_basic(self):
        """Forward returns a scalar (mean reduction)."""
        fn = build_loss_fn("circle_loss", category="metric_learning")
        out = fn(torch.randn(10, 32), torch.randint(0, 3, (10,)))
        assert out.shape == ()
        assert out.item() > 0

    def test_separated_low(self):
        """Well-separated embeddings produce low loss."""
        fn = build_loss_fn("circle_loss", category="metric_learning", margin=0.25, gamma=80.)
        # Each class on a separate axis
        emb = torch.zeros(6, 3)
        emb[0, 0] = 1.0  # class 0
        emb[1, 0] = 1.0
        emb[2, 1] = 1.0  # class 1
        emb[3, 1] = 1.0
        emb[4, 2] = 1.0  # class 2
        emb[5, 2] = 1.0
        out = fn(emb, torch.tensor([0, 0, 1, 1, 2, 2]))
        assert out.item() < 1.0

    def test_gradient(self):
        """Gradient flows through embeddings."""
        fn = build_loss_fn("circle_loss", category="metric_learning")
        emb = torch.randn(8, 32, requires_grad=True)
        fn(emb, torch.randint(0, 3, (8,))).backward()
        assert emb.grad is not None

    def test_forward_sum_reduction(self):
        """Sum reduction works."""
        fn = build_loss_fn("circle_loss", category="metric_learning", reduction="sum")
        out = fn(torch.randn(10, 32), torch.randint(0, 3, (10,)))
        assert out.shape == ()

    def test_forward_none_reduction(self):
        """None reduction returns per-sample losses."""
        fn = build_loss_fn("circle_loss", category="metric_learning", reduction="none")
        out = fn(torch.randn(10, 32), torch.randint(0, 3, (10,)))
        assert out.shape == (10,)

    def test_extra_repr(self):
        """extra_repr includes key parameters."""
        fn = build_loss_fn("circle_loss", category="metric_learning", margin=0.5, gamma=60.)
        r = fn.extra_repr()
        assert "margin" in r and "gamma" in r
