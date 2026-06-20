"""Tests for WingLoss."""

from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import build_loss_fn, SUPPORTED_LOSSES


class TestWingLoss:
    """Test suite for WingLoss."""

    def test_registered(self):
        """WingLoss is registered in SUPPORTED_LOSSES."""
        assert "regression/wing_loss" in SUPPORTED_LOSSES

    def test_build(self):
        """Can build via build_loss_fn."""
        loss_fn = build_loss_fn("wing_loss", category="regression")
        assert loss_fn is not None

    def test_forward_shape(self):
        """Forward returns a scalar (mean reduction)."""
        loss_fn = build_loss_fn("wing_loss", category="regression")
        p = torch.randn(4, 4)
        t = torch.randn(4, 4)
        out = loss_fn(p, t)
        assert out.shape == ()

    def test_zero_error(self):
        """Zero error yields zero loss."""
        loss_fn = build_loss_fn("wing_loss", category="regression")
        loss = loss_fn(torch.tensor([[2.5]]), torch.tensor([[2.5]]))
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), f"Got {loss.item()}"

    def test_small_error_nonlinear(self):
        """Small error (|x| < width) uses the ln nonlinear branch."""
        w = 10.0
        eps = 2.0
        loss_fn = build_loss_fn(
            "wing_loss", category="regression", width=w, epsilon=eps, reduction="none"
        )
        x = torch.tensor([3.0])  # |x| < w
        p = torch.tensor([5.0])
        t = torch.tensor([2.0])
        loss = loss_fn(p, t)
        # wing = w * ln(1 + |x|/eps) = 10 * ln(1 + 3/2) = 10 * ln(2.5) ≈ 9.1629
        expected = w * torch.log(1.0 + x.abs() / eps)
        assert torch.allclose(loss, expected, atol=1e-4), f"Got {loss.item():.4f}"

    def test_large_error_linear(self):
        """Large error (|x| >= width) uses the linear branch."""
        w = 10.0
        eps = 2.0
        loss_fn = build_loss_fn(
            "wing_loss", category="regression", width=w, epsilon=eps, reduction="none"
        )
        x = torch.tensor([12.0])  # |x| >= w
        p = torch.tensor([15.0])
        t = torch.tensor([3.0])
        loss = loss_fn(p, t)
        # C = w - w * ln(1 + w/eps) = 10 - 10*ln(1 + 10/2) = 10 - 10*ln(6) ≈ -7.9178
        C = w - w * float(torch.log(torch.tensor(1.0 + w / eps)))
        # wing = |x| - C = 12 - (-7.9178) = 19.9178
        expected = x.abs() - C
        assert torch.allclose(loss, expected, atol=1e-4), f"Got {loss.item():.4f}"

    def test_continuity_at_width(self):
        """Loss is continuous at |x| = width."""
        w = 10.0
        eps = 2.0
        loss_fn = build_loss_fn(
            "wing_loss", category="regression", width=w, epsilon=eps, reduction="none"
        )
        # Test at |x| = w - ε and |x| = w + ε
        x_near = torch.tensor([w - 0.01])
        p_near = torch.tensor([w - 0.01])
        t_near = torch.zeros_like(p_near)
        loss_near = loss_fn(p_near, t_near)

        x_far = torch.tensor([w + 0.01])
        p_far = torch.tensor([w + 0.01])
        t_far = torch.zeros_like(p_far)
        loss_far = loss_fn(p_far, t_far)

        # Values should be close
        assert torch.allclose(loss_near, loss_far, atol=0.1), (
            f"near={loss_near.item():.6f} far={loss_far.item():.6f}"
        )

    def test_symmetric(self):
        """Loss is symmetric: same for positive and negative error."""
        loss_fn = build_loss_fn(
            "wing_loss", category="regression", width=10.0, epsilon=2.0, reduction="none"
        )
        loss_pos = loss_fn(torch.tensor([[8.0]]), torch.tensor([[3.0]]))  # error = 5
        loss_neg = loss_fn(torch.tensor([[3.0]]), torch.tensor([[8.0]]))  # error = -5
        assert torch.allclose(loss_pos, loss_neg, atol=1e-6), (
            f"pos={loss_pos.item():.6f} neg={loss_neg.item():.6f}"
        )

    def test_reduction_none(self):
        """reduction='none' returns per-element losses."""
        loss_fn = build_loss_fn(
            "wing_loss", category="regression", width=10.0, epsilon=2.0, reduction="none"
        )
        loss = loss_fn(torch.randn(4, 4), torch.randn(4, 4))
        assert loss.shape == (4, 4)

    def test_reduction_sum(self):
        """reduction='sum' returns scalar sum."""
        loss_fn = build_loss_fn(
            "wing_loss", category="regression", width=10.0, epsilon=2.0, reduction="sum"
        )
        p = torch.tensor([[1.0, 2.0, 3.0]])
        t = torch.zeros_like(p)
        loss = loss_fn(p, t)
        # All small errors: wing = w * ln(1 + |x|/eps)
        # x=1: 10*ln(1+0.5)=10*ln(1.5)≈4.0547
        # x=2: 10*ln(1+1.0)=10*ln(2)≈6.9315
        # x=3: 10*ln(1+1.5)=10*ln(2.5)≈9.1629
        expected = 10.0 * torch.log(1.0 + torch.tensor([1.0, 2.0, 3.0]) / 2.0)
        assert torch.allclose(loss, expected.sum(), atol=1e-3), f"Got {loss.item():.4f}"

    def test_gradient_flow(self):
        """Loss is differentiable w.r.t. prediction."""
        loss_fn = build_loss_fn("wing_loss", category="regression", width=10.0, epsilon=2.0)
        p = torch.randn(4, 4, requires_grad=True)
        t = torch.randn(4, 4)
        loss = loss_fn(p, t)
        loss.backward()
        assert p.grad is not None

    def test_extra_repr(self):
        """extra_repr includes reduction, width, and epsilon."""
        loss_fn = build_loss_fn(
            "wing_loss", category="regression", width=12.0, epsilon=3.0
        )
        r = loss_fn.extra_repr()
        assert "reduction" in r
        assert "width" in r
        assert "epsilon" in r
