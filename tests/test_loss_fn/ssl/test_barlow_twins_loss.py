"""Tests for BarlowTwinsLoss."""
from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import build_loss_fn, SUPPORTED_LOSSES


class TestBarlowTwinsLoss:
    """Test suite for BarlowTwinsLoss."""

    def test_registered(self):
        """BarlowTwinsLoss is registered in SUPPORTED_LOSSES."""
        assert "ssl/barlow_twins_loss" in SUPPORTED_LOSSES

    def test_build(self):
        """Can build via build_loss_fn with defaults."""
        fn = build_loss_fn("barlow_twins_loss", category="ssl")
        assert fn is not None
        assert fn.lambd == 0.005

    def test_basic(self):
        """Forward returns a scalar (mean reduction)."""
        fn = build_loss_fn("barlow_twins_loss", category="ssl")
        out = fn(torch.randn(8, 16), torch.randn(8, 16))
        assert out.shape == ()

    def test_identical_embeddings(self):
        """Identical embeddings produce low loss (< 1)."""
        fn = build_loss_fn("barlow_twins_loss", category="ssl")
        e = torch.randn(8, 32)
        out = fn(e, e)
        assert out.item() < 1.0

    def test_gradient(self):
        """Gradient flows through both inputs."""
        fn = build_loss_fn("barlow_twins_loss", category="ssl")
        p = torch.randn(8, 16, requires_grad=True)
        z = torch.randn(8, 16, requires_grad=True)
        out = fn(p, z)
        out.backward()
        assert p.grad is not None
        assert z.grad is not None

    def test_diagonal_ones_offdiagonal_zeros(self):
        """When cross-corr is identity, loss is zero (ignoring lambd)."""
        fn = build_loss_fn("barlow_twins_loss", category="ssl", lambd=0.0)
        # Perfectly correlated features: after z-score, cross-corr = identity
        # Create data where z-scored versions are identical
        p = torch.randn(8, 4)
        out = fn(p, p)
        # loss = sum(1 - 1)^2 + lambd * sum(0)^2 = 0
        assert out.item() < 1e-6

    def test_lambd_effect(self):
        """Higher lambd increases off-diagonal penalty."""
        # Two uncorrelated random tensors
        torch.manual_seed(42)
        p = torch.randn(8, 4)
        z = torch.randn(8, 4)
        low = build_loss_fn("barlow_twins_loss", category="ssl", lambd=0.0)
        high = build_loss_fn("barlow_twins_loss", category="ssl", lambd=10.0)
        assert high(p, z).item() >= low(p, z).item()

    def test_forward_sum_reduction(self):
        """Sum reduction works."""
        fn = build_loss_fn("barlow_twins_loss", category="ssl", reduction="sum")
        out = fn(torch.randn(8, 16), torch.randn(8, 16))
        assert out.shape == ()

    def test_forward_none_reduction(self):
        """None reduction returns a 1D tensor."""
        fn = build_loss_fn("barlow_twins_loss", category="ssl", reduction="none")
        out = fn(torch.randn(8, 16), torch.randn(8, 16))
        assert out.dim() == 1

    def test_extra_repr(self):
        """extra_repr includes lambd and reduction."""
        fn = build_loss_fn("barlow_twins_loss", category="ssl", lambd=0.01)
        r = fn.extra_repr()
        assert "lambd" in r
        assert "reduction" in r
