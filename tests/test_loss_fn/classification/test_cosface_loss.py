"""Tests for CosFaceLoss."""
from __future__ import annotations

import torch
import pytest
from torch.nn import functional as F
from cvnets.loss_fn import build_loss_fn, SUPPORTED_LOSSES


class TestCosFaceLoss:
    """Test suite for CosFaceLoss."""

    def test_registered(self):
        """CosFaceLoss is registered in SUPPORTED_LOSSES."""
        assert "classification/cosface_loss" in SUPPORTED_LOSSES

    def test_build(self):
        """Can build via build_loss_fn."""
        loss_fn = build_loss_fn(
            "cosface_loss", category="classification",
            embed_dim=64, num_classes=10
        )
        assert loss_fn is not None

    def test_forward_shape(self):
        """Forward returns a scalar (mean reduction)."""
        loss_fn = build_loss_fn(
            "cosface_loss", category="classification",
            embed_dim=64, num_classes=10
        )
        embeddings = torch.randn(4, 64)
        targets = torch.randint(0, 10, (4,))
        out = loss_fn(embeddings, targets)
        assert out.shape == ()

    def test_forward_sum_reduction(self):
        """Sum reduction works."""
        loss_fn = build_loss_fn(
            "cosface_loss", category="classification",
            embed_dim=64, num_classes=10, reduction="sum"
        )
        embeddings = torch.randn(4, 64)
        targets = torch.randint(0, 10, (4,))
        out = loss_fn(embeddings, targets)
        assert out.shape == ()

    def test_forward_none_reduction(self):
        """None reduction returns per-sample losses."""
        loss_fn = build_loss_fn(
            "cosface_loss", category="classification",
            embed_dim=64, num_classes=10, reduction="none"
        )
        embeddings = torch.randn(4, 64)
        targets = torch.randint(0, 10, (4,))
        out = loss_fn(embeddings, targets)
        assert out.shape == (4,)

    def test_weight_shape(self):
        """Weight matrix has correct shape (num_classes, embed_dim)."""
        loss_fn = build_loss_fn(
            "cosface_loss", category="classification",
            embed_dim=128, num_classes=20
        )
        assert loss_fn.weight.shape == (20, 128)

    def test_device_and_dtype(self):
        """Weight is float32 and on correct device."""
        loss_fn = build_loss_fn(
            "cosface_loss", category="classification",
            embed_dim=64, num_classes=10
        )
        assert loss_fn.weight.dtype == torch.float32

    def test_margin_and_scale(self):
        """Custom margin and scale are accepted."""
        loss_fn = build_loss_fn(
            "cosface_loss", category="classification",
            embed_dim=64, num_classes=10, margin=0.2, scale=32.0
        )
        embeddings = torch.randn(4, 64)
        targets = torch.randint(0, 10, (4,))
        out = loss_fn(embeddings, targets)
        assert out.shape == ()

    def test_loss_non_negative(self):
        """Loss should be non-negative."""
        loss_fn = build_loss_fn(
            "cosface_loss", category="classification",
            embed_dim=64, num_classes=10
        )
        embeddings = torch.randn(4, 64)
        targets = torch.randint(0, 10, (4,))
        out = loss_fn(embeddings, targets)
        assert out.item() >= 0.0

    def test_margin_zero_equals_ce(self):
        """margin=0 should produce similar loss to standard CE on softmax."""
        cosface_fn = build_loss_fn(
            "cosface_loss", category="classification",
            embed_dim=64, num_classes=10, margin=0.0, scale=1.0
        )
        ce_fn = build_loss_fn("cross_entropy", category="classification")
        embeddings = torch.randn(4, 64)
        targets = torch.randint(0, 10, (4,))
        cf_out = cosface_fn(embeddings.clone(), targets.clone())
        # Convert embeddings to logits via weight matrix
        weight = cosface_fn.weight.detach()
        logits = torch.mm(F.normalize(embeddings.clone()), F.normalize(weight).t())
        ce_out = ce_fn(logits, targets.clone())
        assert cf_out.item() == pytest.approx(ce_out.item(), rel=1e-5)

    def test_extra_repr(self):
        """extra_repr includes key parameters."""
        loss_fn = build_loss_fn(
            "cosface_loss", category="classification",
            embed_dim=64, num_classes=10, margin=0.35, scale=64.0
        )
        r = loss_fn.extra_repr()
        assert "margin" in r
        assert "scale" in r
        assert "embed_dim" in r
        assert "num_classes" in r
