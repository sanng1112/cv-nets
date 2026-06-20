"""
Callback system for the cv-nets training pipeline.

Provides an abstract ``Callback`` base class, a ``CallbackList`` dispatcher,
and built-in callbacks: ``MetricsLogger``, ``ModelCheckpoint``,
``EarlyStopping``, and ``ProgressBar``.
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch

from cvnets.utils.logger import info, log, warning


class Callback(ABC):
    """Abstract base class for all training callbacks."""

    def on_train_start(self, trainer: Any) -> None:
        """Called once before the training loop begins."""

    def on_train_end(self, trainer: Any) -> None:
        """Called once after the training loop ends."""

    def on_epoch_start(self, trainer: Any) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, trainer: Any) -> None:
        """Called at the end of each epoch (after validation)."""

    def on_batch_start(self, trainer: Any) -> None:
        """Called at the start of each training batch."""

    def on_batch_end(self, trainer: Any) -> None:
        """Called at the end of each training batch."""

    def on_validation_start(self, trainer: Any) -> None:
        """Called at the start of the validation loop."""

    def on_validation_end(self, trainer: Any) -> None:
        """Called at the end of the validation loop."""

class CallbackList:
    """Container that dispatches lifecycle events to all registered callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        self._callbacks: List[Callback] = callbacks if callbacks is not None else []

    def add_callback(self, callback: Callback) -> None:
        self._callbacks.append(callback)

    @property
    def callbacks(self) -> List[Callback]:
        return self._callbacks

    def on_train_start(self, trainer: Any) -> None:
        for cb in self._callbacks:
            cb.on_train_start(trainer)

    def on_train_end(self, trainer: Any) -> None:
        for cb in self._callbacks:
            cb.on_train_end(trainer)

    def on_epoch_start(self, trainer: Any) -> None:
        for cb in self._callbacks:
            cb.on_epoch_start(trainer)

    def on_epoch_end(self, trainer: Any) -> None:
        for cb in self._callbacks:
            cb.on_epoch_end(trainer)

    def on_batch_start(self, trainer: Any) -> None:
        for cb in self._callbacks:
            cb.on_batch_start(trainer)

    def on_batch_end(self, trainer: Any) -> None:
        for cb in self._callbacks:
            cb.on_batch_end(trainer)

    def on_validation_start(self, trainer: Any) -> None:
        for cb in self._callbacks:
            cb.on_validation_start(trainer)

    def on_validation_end(self, trainer: Any) -> None:
        for cb in self._callbacks:
            cb.on_validation_end(trainer)


class MetricsLogger(Callback):
    """Print epoch-level metrics at the end of each epoch."""

    def on_epoch_end(self, trainer: Any) -> None:
        metrics = trainer.metrics_tracker.compute()
        parts = [f"{name}: {value:.4f}" for name, value in metrics.items()]
        log(f"Epoch {trainer.current_epoch}/{trainer.num_epochs} - {' | '.join(parts)}")


class ModelCheckpoint(Callback):
    """Save the model when a monitored metric achieves a new best value.

    Args:
        save_dir: Directory where checkpoint files will be written.
        monitor: Name of the metric to monitor (key in the metrics dict).
        mode: ``'max'`` (higher is better) or ``'min'`` (lower is better).
        filename: Template string for the checkpoint filename. May contain
            ``{epoch}``, ``{metric}``, and ``{value}`` placeholders.
    """

    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_loss",
        mode: str = "min",
        filename: str = "checkpoint_epoch_{epoch}.pt",
    ) -> None:
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.filename = filename

        self._best: Optional[float] = None
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, trainer: Any) -> None:
        metrics = trainer.metrics_tracker.compute()
        if self.monitor not in metrics:
            warning(f"Metric '{self.monitor}' not found in {list(metrics.keys())}")
            return

        current = metrics[self.monitor]

        improved: bool
        if self._best is None:
            improved = True
        elif self.mode == "max":
            improved = current > self._best
        else:  # mode == "min"
            improved = current < self._best

        if improved:
            self._best = current
            fname = self.filename.format(
                epoch=trainer.current_epoch,
                metric=self.monitor,
                value=current,
            )
            path = os.path.join(self.save_dir, fname)
            torch.save(trainer.model.state_dict(), path)
            info(f"Checkpoint saved to {path} (monitor={self.monitor}: {current:.4f})")


class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving.

    Args:
        monitor: Name of the metric to monitor.
        patience: Number of epochs with no improvement after which training
            is stopped.
        mode: ``'max'`` or ``'min'``.
        min_delta: Minimum change in the monitored metric to qualify as an
            improvement.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.001,
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self._best: Optional[float] = None
        self._counter: int = 0

    def on_epoch_end(self, trainer: Any) -> None:
        metrics = trainer.metrics_tracker.compute()
        if self.monitor not in metrics:
            warning(f"Metric '{self.monitor}' not found; early stopping disabled.")
            return

        current = metrics[self.monitor]

        if self._best is None:
            self._best = current
            self._counter = 0
            return

        # Determine improvement
        if self.mode == "max":
            delta = current - self._best
        else:
            delta = self._best - current

        if delta > self.min_delta:
            # Improvement
            self._best = current
            self._counter = 0
        else:
            self._counter += 1
            info(
                f"EarlyStopping {self._counter}/{self.patience} "
                f"(best {self.monitor}={self._best:.4f})"
            )
            if self._counter >= self.patience:
                info(f"Early stopping triggered after {trainer.current_epoch} epochs")
                trainer.should_stop = True


class ProgressBar(Callback):
    """Display a ``tqdm`` progress bar for each training epoch."""

    def __init__(self) -> None:
        self._progress_bar: Any = None

    def on_epoch_start(self, trainer: Any) -> None:
        from tqdm import tqdm

        self._progress_bar = tqdm(
            total=len(trainer.train_loader),
            desc=f"Epoch {trainer.current_epoch}/{trainer.num_epochs}",
            unit="batch",
            leave=True,
        )

    def on_batch_end(self, trainer: Any) -> None:
        if self._progress_bar is not None:
            self._progress_bar.update(1)

    def on_epoch_end(self, trainer: Any) -> None:
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
