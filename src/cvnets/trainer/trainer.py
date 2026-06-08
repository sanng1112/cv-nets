"""
Training pipeline for cv-nets.

Provides the ``Trainer`` class that orchestrates model training with
metric tracking, callback dispatch, gradient accumulation, and
optional validation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from cvnets.trainer.metrics import Accuracy, AverageLoss, MetricsTracker
from cvnets.trainer.callbacks import Callback, CallbackList
from cvnets.utils.logger import info, log


class Trainer:
    """Orchestrate model training with metric tracking and callbacks.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader providing training batches.
        optimizer: Optimizer for updating model parameters.
        criterion: Loss function (callable taking ``(output, target)``).
        val_loader: Optional DataLoader for validation.
        num_epochs: Number of full passes over the training data.
        device: Device string (e.g. ``'cuda'``, ``'cpu'``). If ``None``,
            auto-detected.
        callbacks: List of callbacks to attach.
        grad_accum_steps: Number of batches to accumulate gradients over
            before each optimizer step.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        criterion: Callable[..., Tensor],
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        device: Optional[str] = None,
        callbacks: Optional[List[Callback]] = None,
        grad_accum_steps: int = 1,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.grad_accum_steps = grad_accum_steps

        # Device auto-detection
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        # Metrics
        self.metrics_tracker = MetricsTracker(
            Accuracy(),
            AverageLoss(),
            metric_names=["accuracy", "avg_loss"],
        )

        # Callbacks
        self._callback_list = CallbackList(callbacks or [])

        # Internal state
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.should_stop: bool = False

        info(f"Trainer initialized on device: {self.device}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> Dict[str, float]:
        """Run the full training loop.

        Returns:
            Final metrics dictionary.
        """
        info("Starting training...")
        self._callback_list.on_train_start(self)

        for epoch in range(1, self.num_epochs + 1):
            if self.should_stop:
                break
            self.current_epoch = epoch
            self._run_epoch()

        self._callback_list.on_train_end(self)
        metrics = self.metrics_tracker.compute()
        info(f"Training complete. Final metrics: {metrics}")
        return metrics

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _run_epoch(self) -> None:
        """Run a single training epoch (train + optional validation)."""
        self._callback_list.on_epoch_start(self)

        # --- Training phase ---
        self.model.train()
        self.metrics_tracker.reset()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            self._callback_list.on_batch_start(self)

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Scale loss for gradient accumulation
            loss_scaled = loss / self.grad_accum_steps
            loss_scaled.backward()

            # Optimizer step (every grad_accum_steps batches)
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Update metrics
            self.metrics_tracker.on_batch_end(
                prediction=outputs,
                target=targets,
                loss_value=loss.item(),
                batch_size=inputs.size(0),
            )

            self._callback_list.on_batch_end(self)

        # Flush remaining gradients if any
        if (len(self.train_loader) % self.grad_accum_steps) != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1

        # --- Validation phase ---
        if self.val_loader is not None:
            self._run_validation()

        self._callback_list.on_epoch_end(self)

    # ------------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_validation(self) -> None:
        """Evaluate the model on the validation set."""
        self._callback_list.on_validation_start(self)

        self.model.eval()
        val_metrics_tracker = MetricsTracker(
            Accuracy(),
            AverageLoss(),
            metric_names=["accuracy", "avg_loss"],
        )

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            val_metrics_tracker.on_batch_end(
                prediction=outputs,
                target=targets,
                loss_value=loss.item(),
                batch_size=inputs.size(0),
            )

        # Merge validation metrics into the main tracker with "val_" prefix
        for name, metric in val_metrics_tracker._metrics.items():
            val_name = f"val_{name}"
            if val_name in self.metrics_tracker._metrics:
                self.metrics_tracker._metrics[val_name] = metric
            else:
                self.metrics_tracker.add_metric(val_name, metric)

        self._callback_list.on_validation_end(self)

