"""
Training pipeline for cv-nets.

Provides the ``Trainer`` class that orchestrates model training with
metric tracking, callback dispatch, gradient accumulation, mixed
precision (AMP), gradient clipping, LR scheduler support, and
optional validation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from cvnets.trainer.metrics import Accuracy, AverageLoss, MetricsTracker
from cvnets.trainer.callbacks import Callback, CallbackList
from cvnets.utils.logger import info, log


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Any = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Persist a training checkpoint to disk.

    Parameters
    ----------
    path : str
        Filesystem path for the checkpoint.
    model : nn.Module
        Model whose ``state_dict`` is saved.
    optimizer : Optimizer, optional
        Optimizer state dict to include.
    scheduler : optional
        Scheduler state dict to include (must have ``state_dict()``).
    epoch : int, optional
        Current epoch number (stored for resumption).
    metrics : dict, optional
        Current metrics to store.
    """
    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if metrics is not None:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, path)
    info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Any = None,
) -> Dict[str, Any]:
    """Load a training checkpoint from disk and restore state.

    Parameters
    ----------
    path : str
        Filesystem path to the checkpoint.
    model : nn.Module
        Model whose ``state_dict`` is restored.
    optimizer : Optimizer, optional
        If provided, the optimizer state dict is restored.
    scheduler : optional
        If provided, the scheduler state dict is restored.

    Returns
    -------
    dict
        The raw checkpoint dictionary (may contain ``'epoch'``, ``'metrics'``,
        and other user keys).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    info(f"Checkpoint loaded from {path}")
    return checkpoint


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


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
        scheduler: Optional LR scheduler (must have a ``step()`` method).
            Called after each validation phase.
        use_amp: If ``True``, enable ``torch.amp.autocast`` and use a
            ``GradScaler`` for mixed-precision training.
        clip_grad_norm: If > 0, clip gradient norm to this value via
            ``torch.nn.utils.clip_grad_norm_()`` before each optimizer step.
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
        scheduler: Any = None,
        use_amp: bool = False,
        clip_grad_norm: float = 0.0,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.grad_accum_steps = grad_accum_steps
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.clip_grad_norm = clip_grad_norm

        # Device auto-detection
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        # Mixed-precision scaler
        self.scaler: Optional[GradScaler] = None
        if self.use_amp:
            self.scaler = GradScaler()

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

            # Forward pass (optionally with AMP)
            if self.use_amp:
                with torch.amp.autocast(device_type=self.device.type):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # Scale loss for gradient accumulation
            loss_scaled = loss / self.grad_accum_steps

            # Backward pass (optionally with GradScaler)
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # Optimizer step (every grad_accum_steps batches)
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self._optimizer_step()
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
            self._optimizer_step()
            self.optimizer.zero_grad()
            self.global_step += 1

        # --- Validation phase ---
        if self.val_loader is not None:
            self._run_validation()

        # --- LR scheduler step (after validation) ---
        if self.scheduler is not None:
            old_lr = self._get_lr()
            self.scheduler.step()
            new_lr = self._get_lr()
            if old_lr != new_lr:
                info(
                    f"Epoch {self.current_epoch}: LR updated "
                    f"{old_lr:.6f} -> {new_lr:.6f}"
                )

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

    # ------------------------------------------------------------------
    # Optimizer step with optional gradient clipping and AMP
    # ------------------------------------------------------------------

    def _optimizer_step(self) -> None:
        """Perform a single optimizer step with optional gradient clipping
        and AMP unscaling."""
        if self.use_amp and self.scaler is not None:
            # Unscale before clipping
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        if self.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_norm
            )

        if self.use_amp and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_lr(self) -> float:
        """Return the current learning rate from the first param group."""
        for param_group in self.optimizer.param_groups:
            return float(param_group["lr"])
        return 0.0
