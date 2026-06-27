"""
Metrics module for the cv-nets training pipeline.

Provides abstract base ``Metric`` and concrete implementations ``Accuracy``,
``AverageLoss``, as well as a ``MetricsTracker`` container.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor


class Metric(ABC):
    """Abstract base for a single metric tracked during training/validation."""

    @abstractmethod
    def update(self, prediction: Tensor, target: Tensor) -> None:
        """Update internal state with a new batch of predictions and targets."""

    @abstractmethod
    def compute(self) -> float:
        """Return the current metric value aggregated over all seen batches."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state so the metric can be reused for a new epoch."""


class Accuracy(Metric):
    """Top-1 classification accuracy in percent."""

    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        _, predicted = torch.max(prediction, 1)
        self.correct += (predicted == target).sum().item()
        self.total += target.size(0)

    def compute(self) -> float:
        return 100.0 * self.correct / self.total if self.total > 0 else 0.0

    def reset(self) -> None:
        self.correct = 0
        self.total = 0


class AverageLoss(Metric):
    """Weighted average loss over batches (weighted by batch size)."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        """No-op: use ``update_loss`` instead."""
        pass

    def update_loss(self, loss_value: float, batch_size: int) -> None:
        """Accumulate a loss value weighted by *batch_size*."""
        self.total += loss_value * batch_size
        self.count += batch_size

    def compute(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


class MetricsTracker:
    """Container that dispatches batch results to a collection of metrics."""

    def __init__(self, *metrics: Metric, metric_names: Optional[List[str]] = None) -> None:
        self._metrics: Dict[str, Metric] = {}
        if metric_names is None:
            metric_names = [f"metric_{i}" for i in range(len(metrics))]
        for name, metric in zip(metric_names, metrics):
            self._metrics[name] = metric

    def add_metric(self, name: str, metric: Metric) -> None:
        """Register an additional metric under *name*."""
        self._metrics[name] = metric

    @property
    def metrics(self) -> Dict[str, Metric]:
        return self._metrics

    def on_batch_end(
        self,
        prediction: Tensor,
        target: Tensor,
        loss_value: Optional[float] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """Update all metrics with the current batch results.

        Args:
            prediction: Model output logits / predictions.
            target: Ground-truth labels.
            loss_value: Scalar loss for this batch (optional, used by
                ``AverageLoss``).
            batch_size: Number of samples in the batch (required if
                *loss_value* is provided).
        """
        for metric in self._metrics.values():
            metric.update(prediction, target)

        # Update AverageLoss metrics if loss information is available.
        if loss_value is not None and batch_size is not None:
            for metric in self._metrics.values():
                if isinstance(metric, AverageLoss):
                    metric.update_loss(loss_value, batch_size)

    def compute(self) -> Dict[str, float]:
        """Return a dictionary mapping metric name to its current value."""
        return {name: metric.compute() for name, metric in self._metrics.items()}

    def reset(self) -> None:
        """Reset every registered metric."""
        for metric in self._metrics.values():
            metric.reset()
