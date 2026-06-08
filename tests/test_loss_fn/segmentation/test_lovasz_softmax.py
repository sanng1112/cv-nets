"""Tests for LovaszSoftmax."""

from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import build_loss_fn, SUPPORTED_LOSSES


class TestLovaszSoftmax:
    """Test suite for LovaszSoftmax."""

    def test_registered(self):
        """LovaszSoftmax is registered in SUPPORTED_LOSSES."""
        assert "segmentation/lovasz_softmax" in SUPPORTED_LOSSES

    def test_build(self):
        """Can build via build_loss_fn."""
        fn = build_loss_fn("lovasz_softmax", category="segmentation")
        assert fn is not None

    def test_basic(self):
        """Forward returns a positive scalar."""
        fn = build_loss_fn("lovasz_softmax", category="segmentation")
        out = fn(torch.randn(2, 5, 16, 16), torch.randint(0, 5, (2, 16, 16)))
        assert out.shape == ()
        assert out.item() > 0

    def test_perfect_low(self):
        """Perfect prediction yields near-zero loss."""
        fn = build_loss_fn("lovasz_softmax", category="segmentation")
        p = torch.full((2, 3, 8, 8), -100.0)
        p[:, 0, :, :] = 100.0
        loss = fn(p, torch.zeros(2, 8, 8, dtype=torch.long))
        assert loss.item() < 0.05

    def test_gradient(self):
        """Backward produces non-None gradient."""
        fn = build_loss_fn("lovasz_softmax", category="segmentation")
        p = torch.randn(2, 3, 8, 8, requires_grad=True)
        fn(p, torch.randint(0, 3, (2, 8, 8))).backward()
        assert p.grad is not None
        assert p.grad.shape == p.shape

    def test_forward_sum_reduction(self):
        """Sum reduction works."""
        fn = build_loss_fn("lovasz_softmax", category="segmentation", reduction="sum")
        out = fn(torch.randn(2, 5, 16, 16), torch.randint(0, 5, (2, 16, 16)))
        assert out.shape == ()

    def test_forward_none_reduction(self):
        """None reduction returns per-batch losses."""
        fn = build_loss_fn(
            "lovasz_softmax", category="segmentation", reduction="none"
        )
        out = fn(torch.randn(2, 5, 16, 16), torch.randint(0, 5, (2, 16, 16)))
        assert out.shape == (2,)

    def test_extra_repr(self):
        """extra_repr includes reduction."""
        fn = build_loss_fn("lovasz_softmax", category="segmentation")
        r = fn.extra_repr()
        assert "reduction" in r
