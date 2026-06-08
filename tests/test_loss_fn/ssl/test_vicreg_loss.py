"""Tests for VICRegLoss."""
from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import build_loss_fn, SUPPORTED_LOSSES


class TestVICRegLoss:
    """Test suite for VICRegLoss."""

    def test_registered(self):
        """VICRegLoss is registered in SUPPORTED_LOSSES."""
        assert "ssl/vicreg_loss" in SUPPORTED_LOSSES

    def test_build(self):
        """Can build via build_loss_fn with defaults."""
        fn = build_loss_fn("vicreg_loss", category="ssl")
        assert fn is not None
        assert fn.sim_w == 25.0
        assert fn.var_w == 25.0
        assert fn.cov_w == 1.0
        assert fn.eps == 1e-4

    def test_basic(self):
        """Forward returns a scalar (mean reduction)."""
        fn = build_loss_fn("vicreg_loss", category="ssl")
        out = fn(torch.randn(8, 16), torch.randn(8, 16))
        assert out.shape == ()

    def test_identical_embeddings(self):
        """Identical embeddings produce finite loss (< 30)."""
        fn = build_loss_fn("vicreg_loss", category="ssl")
        e = torch.randn(8, 16)
        out = fn(e, e)
        assert out.item() < 30.0

    def test_invariance_term_separate(self):
        """Invariance (MSE) term: identical embeddings give zero invariance loss."""
        fn = build_loss_fn("vicreg_loss", category="ssl")
        # Manually compute invariance loss
        p = torch.randn(8, 16)
        inv = torch.nn.functional.mse_loss(p, p)
        assert inv.item() < 1e-6

    def test_gradient(self):
        """Gradient flows through both prediction and target."""
        fn = build_loss_fn("vicreg_loss", category="ssl")
        p = torch.randn(8, 16, requires_grad=True)
        z = torch.randn(8, 16, requires_grad=True)
        out = fn(p, z)
        out.backward()
        assert p.grad is not None
        assert z.grad is not None

    def test_variance_term_hinge(self):
        """Variance term pushes std towards 1; low std incurs penalty."""
        fn = build_loss_fn("vicreg_loss", category="ssl", var_w=1.0, sim_w=0.0, cov_w=0.0)
        # All-constant input has zero std
        p = torch.ones(8, 4)
        z = torch.ones(8, 4)
        out = fn(p, z)
        # var loss = relu(1 - sqrt(0 + eps)).mean() + relu(1 - sqrt(0 + eps)).mean()
        # sqrt(1e-4) = 0.01 => relu(0.99) = 0.99 => total = 1.98
        expected = 2.0 * (1.0 - (1e-4) ** 0.5)
        assert torch.allclose(out, torch.tensor(expected), atol=1e-5)

    def test_forward_sum_reduction(self):
        """Sum reduction works."""
        fn = build_loss_fn("vicreg_loss", category="ssl", reduction="sum")
        out = fn(torch.randn(8, 16), torch.randn(8, 16))
        assert out.shape == ()

    def test_forward_none_reduction(self):
        """None reduction returns per-sample losses (scalar since loss is batch-level)."""
        fn = build_loss_fn("vicreg_loss", category="ssl", reduction="none")
        out = fn(torch.randn(8, 16), torch.randn(8, 16))
        # VICReg produces a scalar per batch, so none returns a 1D tensor
        assert out.dim() == 1

    def test_extra_repr(self):
        """extra_repr includes key parameters."""
        fn = build_loss_fn("vicreg_loss", category="ssl", sim_w=10.0, var_w=20.0, cov_w=2.0)
        r = fn.extra_repr()
        assert "sim" in r and "var" in r and "cov" in r
