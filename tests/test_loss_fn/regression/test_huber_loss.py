"""Tests for HuberLoss."""

from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import build_loss_fn, SUPPORTED_LOSSES


class TestHuberLoss:
    """Test suite for HuberLoss."""

    def test_registered(self):
        """HuberLoss is registered in SUPPORTED_LOSSES."""
        assert "regression/huber_loss" in SUPPORTED_LOSSES

    def test_build(self):
        """Can build via build_loss_fn."""
        loss_fn = build_loss_fn("huber_loss", category="regression")
        assert loss_fn is not None

    def test_forward_shape(self):
        """Forward returns a scalar (mean reduction)."""
        loss_fn = build_loss_fn("huber_loss", category="regression")
        p = torch.randn(4, 4)
        t = torch.randn(4, 4)
        out = loss_fn(p, t)
        assert out.shape == ()

    def test_matches_f_huber_loss(self):
        """Our wrapper matches torch.nn.functional.huber_loss output."""
        from torch.nn import functional as F
        loss_fn = build_loss_fn("huber_loss", category="regression", delta=1.0)
        p = torch.randn(8, 3)
        t = torch.randn(8, 3)
        out = loss_fn(p, t)
        expected = F.huber_loss(p, t, reduction="mean")
        assert torch.allclose(out, expected, atol=1e-6)

    def test_small_error_quadratic(self):
        """Small error (|x| <= delta) uses quadratic branch."""
        loss_fn = build_loss_fn("huber_loss", category="regression", delta=1.0)
        p = torch.tensor([[0.5]])
        t = torch.tensor([[0.0]])
        loss = loss_fn(p, t)
        # 0.5 * (0.5)^2 = 0.125
        assert torch.allclose(loss, torch.tensor(0.125), atol=1e-4), f"Got {loss.item()}"

    def test_large_error_linear(self):
        """Large error (|x| > delta) uses linear branch."""
        loss_fn = build_loss_fn("huber_loss", category="regression", delta=1.0)
        p = torch.tensor([[5.0]])
        t = torch.tensor([[0.0]])
        loss = loss_fn(p, t)
        # delta * (|x| - 0.5 * delta) = 1.0 * (5.0 - 0.5) = 4.5
        assert torch.allclose(loss, torch.tensor(4.5), atol=1e-4), f"Got {loss.item()}"

    def test_custom_delta(self):
        """Custom delta value changes transition point."""
        loss_fn = build_loss_fn("huber_loss", category="regression", delta=2.0)
        # |diff| = 1.0 < delta=2.0 → quadratic: 0.5 * (1.0)^2 = 0.5
        p = torch.tensor([[1.0]])
        t = torch.tensor([[0.0]])
        loss = loss_fn(p, t)
        assert torch.allclose(loss, torch.tensor(0.5), atol=1e-4), f"Got {loss.item()}"

    def test_reduction_none(self):
        """reduction='none' returns per-element losses."""
        loss_fn = build_loss_fn("huber_loss", category="regression", delta=1.0, reduction="none")
        loss = loss_fn(torch.randn(4, 4), torch.randn(4, 4))
        assert loss.shape == (4, 4)

    def test_reduction_sum(self):
        """reduction='sum' returns scalar sum."""
        loss_fn = build_loss_fn("huber_loss", category="regression", delta=1.0, reduction="sum")
        p = torch.tensor([[1.0, 2.0, 3.0]])
        t = torch.zeros_like(p)
        loss = loss_fn(p, t)
        # |diff| > 1, so huber = delta * (|diff| - 0.5*delta) = 1.0 * (|diff| - 0.5)
        # sum = (1-0.5) + (2-0.5) + (3-0.5) = 0.5 + 1.5 + 2.5 = 4.5
        assert torch.allclose(loss, torch.tensor(4.5), atol=1e-4), f"Got {loss.item()}"

    def test_gradient_flow(self):
        """Loss is differentiable w.r.t. prediction."""
        loss_fn = build_loss_fn("huber_loss", category="regression", delta=1.0)
        p = torch.randn(4, 4, requires_grad=True)
        t = torch.randn(4, 4)
        loss = loss_fn(p, t)
        loss.backward()
        assert p.grad is not None

    def test_extra_repr(self):
        """extra_repr includes reduction and delta."""
        loss_fn = build_loss_fn("huber_loss", category="regression", delta=2.5)
        r = loss_fn.extra_repr()
        assert "reduction" in r
        assert "delta" in r
