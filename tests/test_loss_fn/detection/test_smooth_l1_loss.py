"""Tests for Smooth L1 Loss."""

from __future__ import annotations

import torch
from cvnets.loss_fn import build_loss_fn


class TestSmoothL1Loss:
    """Test SmoothL1Loss for detection."""

    def test_basic(self):
        """Basic forward pass with random inputs produces scalar output."""
        fn = build_loss_fn("smooth_l1_loss", category="detection")
        out = fn(torch.randn(4, 4), torch.randn(4, 4))
        assert out.shape == ()

    def test_large_error_linear(self):
        """Large error (|x| > beta) uses L1 branch: loss ≈ |x| - 0.5."""
        fn = build_loss_fn("smooth_l1_loss", category="detection", beta=1.0)
        loss = fn(torch.tensor([[1000.0]]), torch.tensor([[0.0]]))
        # For large diff: smooth_l1 = |diff| - 0.5 = 999.5
        assert torch.allclose(loss, torch.tensor(999.5), atol=1.0), f"Got {loss.item()}"

    def test_small_error_quadratic(self):
        """Small error (|x| <= beta) uses L2 branch: loss ≈ 0.5 * x^2 / beta."""
        fn = build_loss_fn("smooth_l1_loss", category="detection", beta=1.0)
        loss = fn(torch.tensor([[0.5]]), torch.tensor([[0.0]]))
        # 0.5 * (0.5)^2 / 1.0 = 0.125
        assert torch.allclose(loss, torch.tensor(0.125), atol=1e-4), f"Got {loss.item()}"

    def test_custom_beta(self):
        """Custom beta value changes the transition point."""
        fn = build_loss_fn("smooth_l1_loss", category="detection", beta=2.0)
        # |diff| = 1.0 < beta=2.0 → quadratic: 0.5 * 1.0^2 / 2.0 = 0.25
        loss = fn(torch.tensor([[1.0]]), torch.tensor([[0.0]]))
        assert torch.allclose(loss, torch.tensor(0.25), atol=1e-4), f"Got {loss.item()}"

    def test_reduction_none(self):
        """reduction='none' returns per-element losses."""
        fn = build_loss_fn(
            "smooth_l1_loss", category="detection", beta=1.0, reduction="none"
        )
        loss = fn(torch.randn(4, 4), torch.randn(4, 4))
        assert loss.shape == (4, 4)

    def test_reduction_sum(self):
        """reduction='sum' returns scalar sum."""
        fn = build_loss_fn(
            "smooth_l1_loss", category="detection", beta=1.0, reduction="sum"
        )
        p = torch.tensor([[1.0, 2.0, 3.0]])
        t = torch.zeros_like(p)
        loss = fn(p, t)
        # Manual check: each element has |diff| > 1, so smooth_l1 = |diff| - 0.5
        # sum = (1-0.5) + (2-0.5) + (3-0.5) = 0.5 + 1.5 + 2.5 = 4.5
        assert torch.allclose(loss, torch.tensor(4.5), atol=1e-4), f"Got {loss.item()}"

    def test_gradient_flow(self):
        """Loss is differentiable w.r.t. prediction."""
        fn = build_loss_fn("smooth_l1_loss", category="detection", beta=1.0)
        p = torch.randn(4, 4, requires_grad=True)
        t = torch.randn(4, 4)
        loss = fn(p, t)
        loss.backward()
        assert p.grad is not None
