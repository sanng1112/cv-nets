"""Tests for CrossEntropyLoss."""
from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import build_loss_fn, SUPPORTED_LOSSES


class TestCrossEntropyLoss:
    """Test suite for CrossEntropyLoss."""

    def test_registered(self):
        """CrossEntropyLoss is registered in SUPPORTED_LOSSES."""
        assert "classification/cross_entropy" in SUPPORTED_LOSSES

    def test_build(self):
        """Can build via build_loss_fn."""
        loss_fn = build_loss_fn("cross_entropy", category="classification")
        assert loss_fn is not None

    def test_forward_shape(self):
        """Forward returns a scalar (mean reduction)."""
        loss_fn = build_loss_fn("cross_entropy", category="classification")
        p = torch.randn(4, 10)
        t = torch.randint(0, 10, (4,))
        out = loss_fn(p, t)
        assert out.shape == ()

    def test_forward_sum_reduction(self):
        """Sum reduction works."""
        loss_fn = build_loss_fn("cross_entropy", category="classification", reduction="sum")
        p = torch.randn(4, 10)
        t = torch.randint(0, 10, (4,))
        out = loss_fn(p, t)
        assert out.shape == ()

    def test_forward_none_reduction(self):
        """None reduction returns per-sample losses."""
        loss_fn = build_loss_fn("cross_entropy", category="classification", reduction="none")
        p = torch.randn(4, 10)
        t = torch.randint(0, 10, (4,))
        out = loss_fn(p, t)
        assert out.shape == (4,)

    def test_label_smoothing(self):
        """Label smoothing parameter is accepted."""
        loss_fn = build_loss_fn("cross_entropy", category="classification", label_smoothing=0.1)
        p = torch.randn(4, 10)
        t = torch.randint(0, 10, (4,))
        out = loss_fn(p, t)
        assert out.shape == ()

    def test_ignore_index(self):
        """ignore_index is handled."""
        loss_fn = build_loss_fn(
            "cross_entropy", category="classification", ignore_index=-1, reduction="none"
        )
        p = torch.randn(5, 10)
        t = torch.tensor([0, 1, -1, 3, 4])
        out = loss_fn(p, t)
        # The -1 should be ignored (loss = 0 for that sample)
        assert out.shape == (5,)
        assert out[2].item() == pytest.approx(0.0)

    def test_class_weight(self):
        """class_weight parameter is accepted."""
        weight = torch.tensor([0.5, 1.0, 0.5] + [1.0] * 7)
        loss_fn = build_loss_fn(
            "cross_entropy", category="classification", class_weight=weight
        )
        p = torch.randn(4, 10)
        t = torch.randint(0, 10, (4,))
        out = loss_fn(p, t)
        assert out.shape == ()

    def test_extra_repr(self):
        """extra_repr includes key parameters."""
        loss_fn = build_loss_fn(
            "cross_entropy", category="classification", label_smoothing=0.2, ignore_index=-1
        )
        r = loss_fn.extra_repr()
        assert "label_smoothing" in r
        assert "ignore_index" in r
