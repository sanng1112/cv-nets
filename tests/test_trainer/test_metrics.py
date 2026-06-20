"""
Tests for the metrics module.

Covers ``Accuracy``, ``AverageLoss``, and ``MetricsTracker``.
"""

from __future__ import annotations

import torch

from cvnets.trainer.metrics import Accuracy, AverageLoss, MetricsTracker


class TestAccuracy:
    """Test suite for ``Accuracy``."""

    def test_perfect_accuracy(self) -> None:
        """All predictions match targets -> 100% accuracy."""
        acc = Accuracy()
        pred = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        target = torch.tensor([0, 1, 0])
        acc.update(pred, target)
        assert acc.compute() == 100.0

    def test_half_accuracy(self) -> None:
        """Half of predictions match targets -> 50% accuracy."""
        acc = Accuracy()
        pred = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        target = torch.tensor([0, 1, 1, 0])
        acc.update(pred, target)
        assert acc.compute() == 50.0

    def test_reset(self) -> None:
        """After reset, accuracy should be 0.0."""
        acc = Accuracy()
        pred = torch.tensor([[1.0, 0.0]])
        target = torch.tensor([0])
        acc.update(pred, target)
        assert acc.compute() == 100.0
        acc.reset()
        assert acc.compute() == 0.0


class TestAverageLoss:
    """Test suite for ``AverageLoss``."""

    def test_basic_average(self) -> None:
        """Weighted average of loss values."""
        avg = AverageLoss()
        avg.update_loss(1.0, 2)
        avg.update_loss(3.0, 1)
        # total = 1.0*2 + 3.0*1 = 5.0; count = 3; avg = 5.0/3 ≈ 1.6667
        assert abs(avg.compute() - 5.0 / 3.0) < 1e-6

    def test_reset(self) -> None:
        """After reset, average should be 0.0."""
        avg = AverageLoss()
        avg.update_loss(2.0, 2)
        assert avg.compute() == 2.0
        avg.reset()
        assert avg.compute() == 0.0


class TestMetricsTracker:
    """Test suite for ``MetricsTracker``."""

    def test_compute_multiple_metrics(self) -> None:
        """Tracker should return correct values for all metrics."""
        tracker = MetricsTracker(
            Accuracy(),
            AverageLoss(),
            metric_names=["acc", "loss"],
        )

        pred = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        target = torch.tensor([0, 1])
        tracker.on_batch_end(pred, target, loss_value=0.5, batch_size=2)

        results = tracker.compute()
        assert "acc" in results
        assert "loss" in results
        assert results["acc"] == 100.0
        assert abs(results["loss"] - 0.5) < 1e-6

    def test_reset(self) -> None:
        """After reset, metrics should return to initial state."""
        tracker = MetricsTracker(
            Accuracy(),
            metric_names=["acc"],
        )
        pred = torch.tensor([[1.0, 0.0]])
        target = torch.tensor([0])
        tracker.on_batch_end(pred, target)
        assert tracker.compute()["acc"] == 100.0
        tracker.reset()
        assert tracker.compute()["acc"] == 0.0
