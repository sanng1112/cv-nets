"""Tests for NTXentLoss."""
from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import build_loss_fn, SUPPORTED_LOSSES


class TestNTXentLoss:
    """Test suite for NTXentLoss (InfoNCE / NT-Xent)."""

    def test_registered(self):
        """NTXentLoss is registered in SUPPORTED_LOSSES."""
        assert "metric_learning/ntxent_loss" in SUPPORTED_LOSSES

    def test_build(self):
        """Can build via build_loss_fn."""
        fn = build_loss_fn("ntxent_loss", category="metric_learning")
        assert fn is not None

    @staticmethod
    def _make_targets(B: int, num_pos_pairs: int) -> Tensor:
        """Create target vector with repeated indices for positive pairs.

        Simulates a SimCLR-style batch where each sample has one positive
        counterpart.
        """
        assert B % (2 * num_pos_pairs) == 0
        reps = B // (2 * num_pos_pairs)
        ids = torch.arange(num_pos_pairs).repeat_interleave(2 * reps)
        return ids[:B]

    def test_basic(self):
        """Forward returns a positive scalar (mean reduction)."""
        fn = build_loss_fn("ntxent_loss", category="metric_learning")
        out = fn(torch.randn(8, 64), self._make_targets(8, 2))
        assert out.shape == ()
        assert out.item() > 0

    def test_temp_effect(self):
        """Lower temperature yields higher loss (sharper distribution)."""
        hi = build_loss_fn("ntxent_loss", category="metric_learning", temperature=0.1)
        lo = build_loss_fn("ntxent_loss", category="metric_learning", temperature=1.0)
        torch.manual_seed(42)
        e = torch.randn(8, 16)
        i = self._make_targets(8, 2)
        assert hi(e, i).item() > lo(e, i).item()

    def test_gradient(self):
        """Gradient flows through embeddings."""
        fn = build_loss_fn("ntxent_loss", category="metric_learning")
        e = torch.randn(8, 32, requires_grad=True)
        fn(e, self._make_targets(8, 2)).backward()
        assert e.grad is not None

    def test_identical_embeddings(self):
        """When all embeddings are identical, loss is high."""
        fn = build_loss_fn("ntxent_loss", category="metric_learning", temperature=0.5)
        e = torch.ones(8, 32)  # all same
        out = fn(e, self._make_targets(8, 2))
        assert out.item() > 0.5

    def test_forward_sum_reduction(self):
        """Sum reduction works."""
        fn = build_loss_fn("ntxent_loss", category="metric_learning", reduction="sum")
        out = fn(torch.randn(8, 64), self._make_targets(8, 2))
        assert out.shape == ()

    def test_forward_none_reduction(self):
        """None reduction returns per-sample losses."""
        fn = build_loss_fn("ntxent_loss", category="metric_learning", reduction="none")
        out = fn(torch.randn(8, 64), self._make_targets(8, 2))
        assert out.shape == (8,)

    def test_extra_repr(self):
        """extra_repr includes key parameters."""
        fn = build_loss_fn("ntxent_loss", category="metric_learning", temperature=0.3)
        r = fn.extra_repr()
        assert "temperature" in r

    def test_negative_temperature_raises(self):
        """Temperature <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            build_loss_fn("ntxent_loss", category="metric_learning", temperature=-0.1)
