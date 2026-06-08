"""Tests for QuantileLoss."""

from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import build_loss_fn, SUPPORTED_LOSSES


class TestQuantileLoss:
    """Test suite for QuantileLoss."""

    def test_registered(self):
        """QuantileLoss is registered in SUPPORTED_LOSSES."""
        assert "regression/quantile_loss" in SUPPORTED_LOSSES

    def test_build(self):
        """Can build via build_loss_fn."""
        loss_fn = build_loss_fn("quantile_loss", category="regression")
        assert loss_fn is not None

    def test_forward_shape(self):
        """Forward returns a scalar (mean reduction)."""
        loss_fn = build_loss_fn("quantile_loss", category="regression")
        p = torch.randn(4, 4)
        t = torch.randn(4, 4)
        out = loss_fn(p, t)
        assert out.shape == ()

    def test_quantile_0_5_symmetric(self):
        """Quantile=0.5 gives symmetric loss: equal penalty for over/under."""
        loss_fn = build_loss_fn("quantile_loss", category="regression", quantile=0.5)
        # over-predict by 2.0
        loss_over = loss_fn(torch.tensor([[3.0]]), torch.tensor([[1.0]]))
        # under-predict by 2.0
        loss_under = loss_fn(torch.tensor([[1.0]]), torch.tensor([[3.0]]))
        assert torch.allclose(loss_over, loss_under, atol=1e-6), (
            f"over={loss_over.item():.4f} under={loss_under.item():.4f}"
        )

    def test_quantile_0_9_penalizes_under_forecast(self):
        """Quantile=0.9 penalises under-prediction more than over-prediction."""
        loss_fn = build_loss_fn("quantile_loss", category="regression", quantile=0.9)
        # under-predict by 2.0 (target=3, pred=1)
        loss_under = loss_fn(torch.tensor([[1.0]]), torch.tensor([[3.0]]))
        # over-predict by 2.0 (target=1, pred=3)
        loss_over = loss_fn(torch.tensor([[3.0]]), torch.tensor([[1.0]]))
        assert loss_under.item() > loss_over.item(), (
            f"under={loss_under.item():.4f} over={loss_over.item():.4f}"
        )

    def test_quantile_0_1_penalizes_over_forecast(self):
        """Quantile=0.1 penalises over-prediction more than under-prediction."""
        loss_fn = build_loss_fn("quantile_loss", category="regression", quantile=0.1)
        # over-predict by 2.0
        loss_over = loss_fn(torch.tensor([[3.0]]), torch.tensor([[1.0]]))
        # under-predict by 2.0
        loss_under = loss_fn(torch.tensor([[1.0]]), torch.tensor([[3.0]]))
        assert loss_over.item() > loss_under.item(), (
            f"over={loss_over.item():.4f} under={loss_under.item():.4f}"
        )

    def test_formula_manual(self):
        """Manual calculation matches known quantile loss formula."""
        q = 0.3
        loss_fn = build_loss_fn("quantile_loss", category="regression", quantile=q, reduction="none")
        p = torch.tensor([2.0])
        t = torch.tensor([5.0])  # error = -3 (p < t, under-prediction)
        loss = loss_fn(p, t)
        # For under-prediction: loss = q * (t - p) = 0.3 * 3 = 0.9
        expected = q * (t - p)
        assert torch.allclose(loss, expected, atol=1e-6), f"Got {loss.item()}"

    def test_formula_manual_over(self):
        """Manual calculation for over-prediction."""
        q = 0.3
        loss_fn = build_loss_fn("quantile_loss", category="regression", quantile=q, reduction="none")
        p = torch.tensor([5.0])
        t = torch.tensor([2.0])  # error = 3 (p > t, over-prediction)
        loss = loss_fn(p, t)
        # For over-prediction: loss = (q - 1) * (t - p) = -0.7 * -3 = 2.1
        expected = (q - 1) * (t - p)
        assert torch.allclose(loss, expected, atol=1e-6), f"Got {loss.item()}"

    def test_zero_error(self):
        """Zero error yields zero loss."""
        loss_fn = build_loss_fn("quantile_loss", category="regression", quantile=0.3)
        loss = loss_fn(torch.tensor([[2.5]]), torch.tensor([[2.5]]))
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), f"Got {loss.item()}"

    def test_reduction_none(self):
        """reduction='none' returns per-element losses."""
        loss_fn = build_loss_fn("quantile_loss", category="regression", quantile=0.5, reduction="none")
        loss = loss_fn(torch.randn(4, 4), torch.randn(4, 4))
        assert loss.shape == (4, 4)

    def test_reduction_sum(self):
        """reduction='sum' returns scalar sum."""
        loss_fn = build_loss_fn("quantile_loss", category="regression", quantile=0.5, reduction="sum")
        p = torch.tensor([[1.0, 2.0, 3.0]])
        t = torch.zeros_like(p)
        loss = loss_fn(p, t)
        # q=0.5, over-prediction: (q-1)*(t-p) = -0.5 * (-1, -2, -3) = 0.5, 1.0, 1.5 → sum = 3.0
        assert torch.allclose(loss, torch.tensor(3.0), atol=1e-4), f"Got {loss.item()}"

    def test_gradient_flow(self):
        """Loss is differentiable w.r.t. prediction."""
        loss_fn = build_loss_fn("quantile_loss", category="regression", quantile=0.5)
        p = torch.randn(4, 4, requires_grad=True)
        t = torch.randn(4, 4)
        loss = loss_fn(p, t)
        loss.backward()
        assert p.grad is not None

    def test_extra_repr(self):
        """extra_repr includes reduction and quantile."""
        loss_fn = build_loss_fn("quantile_loss", category="regression", quantile=0.9)
        r = loss_fn.extra_repr()
        assert "reduction" in r
        assert "quantile" in r
