"""
Tests for the enhanced Trainer features: scheduler, AMP, gradient
clipping, save_checkpoint / load_checkpoint.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pytest
import torch
from torch import Tensor, nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from cvnets.scheduler import build_scheduler
from cvnets.trainer.trainer import (
    Trainer,
    load_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleLinearModel(nn.Module):
    """A tiny linear model for testing."""

    def __init__(self, in_features: int = 4, out_features: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


def _make_dummy_data(
    n_samples: int = 16, in_features: int = 4, num_classes: int = 2
) -> Tuple[Tensor, Tensor]:
    x = torch.randn(n_samples, in_features)
    y = torch.randint(0, num_classes, (n_samples,))
    return x, y


def _make_dataloader(
    x: Tensor, y: Tensor, batch_size: int = 4
) -> DataLoader[Tuple[Tensor, Tensor]]:
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)


def _make_trainer(
    num_epochs: int = 2,
    **kwargs: Any,
) -> Trainer:
    """Helper to create a Trainer with default dummy data."""
    model = SimpleLinearModel()
    train_loader = _make_dataloader(*_make_dummy_data(16))
    val_loader = _make_dataloader(*_make_dummy_data(8))
    optimizer = SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    return Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device="cpu",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests - Scheduler
# ---------------------------------------------------------------------------


class TestTrainerScheduler:
    """Tests for scheduler support in Trainer."""

    def test_trainer_with_scheduler(self) -> None:
        """Trainer should accept a scheduler and decay LR."""
        model = SimpleLinearModel()
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = build_scheduler(optimizer, "step", step_size=1, gamma=0.5)
        trainer = Trainer(
            model=model,
            train_loader=_make_dataloader(*_make_dummy_data(16)),
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            val_loader=_make_dataloader(*_make_dummy_data(8)),
            num_epochs=3,
            device="cpu",
            scheduler=scheduler,
        )
        initial_lr = optimizer.param_groups[0]["lr"]
        trainer.fit()
        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr

    def test_scheduler_step_called(self) -> None:
        """Verify scheduler.step() decays LR appropriately."""
        model = SimpleLinearModel()
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = build_scheduler(optimizer, "step", step_size=1, gamma=0.5)
        trainer = Trainer(
            model=model,
            train_loader=_make_dataloader(*_make_dummy_data(16)),
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            val_loader=_make_dataloader(*_make_dummy_data(8)),
            num_epochs=3,
            device="cpu",
            scheduler=scheduler,
        )
        trainer.fit()
        expected_lr = 0.1 * (0.5 ** 3)
        assert math.isclose(optimizer.param_groups[0]["lr"], expected_lr, rel_tol=1e-5)

    def test_trainer_no_scheduler(self) -> None:
        """Trainer should work without scheduler."""
        trainer = _make_trainer()
        metrics = trainer.fit()
        assert "accuracy" in metrics
        assert "avg_loss" in metrics


# ---------------------------------------------------------------------------
# Tests - AMP
# ---------------------------------------------------------------------------


class TestTrainerAMP:
    """Tests for mixed-precision (AMP) support in Trainer."""

    def test_trainer_with_amp(self) -> None:
        """Trainer should accept use_amp=True and train without error."""
        trainer = _make_trainer(use_amp=True)
        assert trainer.use_amp is True
        assert trainer.scaler is not None
        metrics = trainer.fit()
        assert "accuracy" in metrics

    def test_trainer_amp_off_by_default(self) -> None:
        """Trainer should have AMP disabled by default."""
        trainer = _make_trainer()
        assert trainer.use_amp is False
        assert trainer.scaler is None

    def test_trainer_amp_with_validation(self) -> None:
        """Trainer with AMP and validation should work."""
        trainer = _make_trainer(use_amp=True, num_epochs=2)
        metrics = trainer.fit()
        assert "val_accuracy" in metrics
        assert "val_avg_loss" in metrics


# ---------------------------------------------------------------------------
# Tests - Gradient Clipping
# ---------------------------------------------------------------------------


class TestTrainerGradientClipping:
    """Tests for gradient clipping support in Trainer."""

    def test_trainer_with_clip_grad_norm(self) -> None:
        """Trainer should accept clip_grad_norm and train without error."""
        trainer = _make_trainer(clip_grad_norm=1.0)
        assert trainer.clip_grad_norm == 1.0
        metrics = trainer.fit()
        assert "accuracy" in metrics

    def test_trainer_clip_grad_norm_zero_by_default(self) -> None:
        """Trainer should have clip_grad_norm=0 by default (disabled)."""
        trainer = _make_trainer()
        assert trainer.clip_grad_norm == 0.0

    def test_trainer_clip_grad_norm_with_scheduler(self) -> None:
        """Trainer with clipping and scheduler should work together."""
        model = SimpleLinearModel()
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = build_scheduler(optimizer, "step", step_size=1, gamma=0.5)
        trainer = Trainer(
            model=model,
            train_loader=_make_dataloader(*_make_dummy_data(16)),
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            val_loader=_make_dataloader(*_make_dummy_data(8)),
            num_epochs=2,
            device="cpu",
            scheduler=scheduler,
            clip_grad_norm=5.0,
        )
        metrics = trainer.fit()
        assert "accuracy" in metrics
        assert "val_accuracy" in metrics


# ---------------------------------------------------------------------------
# Tests - Checkpoint
# ---------------------------------------------------------------------------


class TestTrainerCheckpoint:
    """Tests for save_checkpoint and load_checkpoint."""

    def test_save_checkpoint_basic(self, tmp_path: Path) -> None:
        """Save a checkpoint with model only."""
        model = SimpleLinearModel()
        path = str(tmp_path / "checkpoint.pt")
        save_checkpoint(path, model=model)
        assert Path(path).exists()

    def test_save_and_load_checkpoint(self, tmp_path: Path) -> None:
        """Save and load a checkpoint with model + optimizer."""
        model = SimpleLinearModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        path = str(tmp_path / "checkpoint.pt")

        # Train one step to change weights
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        original_weights = model.fc.weight.clone()

        save_checkpoint(path, model=model, optimizer=optimizer, epoch=1)

        # Load into a new model
        new_model = SimpleLinearModel()
        new_optimizer = SGD(new_model.parameters(), lr=0.01)
        checkpoint = load_checkpoint(path, model=new_model, optimizer=new_optimizer)

        assert checkpoint["epoch"] == 1
        assert torch.allclose(new_model.fc.weight, original_weights)

    def test_save_and_load_with_scheduler(self, tmp_path: Path) -> None:
        """Save and load a checkpoint with scheduler."""
        model = SimpleLinearModel()
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = build_scheduler(optimizer, "step", step_size=1, gamma=0.5)
        path = str(tmp_path / "checkpoint.pt")

        # Step optimizer before scheduler to avoid PyTorch warning
        optimizer.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        scheduler.step()

        save_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler, epoch=2
        )

        # Load into a fresh model/optimizer/scheduler
        new_model = SimpleLinearModel()
        new_optimizer = SGD(new_model.parameters(), lr=0.1)
        new_scheduler = build_scheduler(new_optimizer, "step", step_size=1, gamma=0.5)

        checkpoint = load_checkpoint(
            path, model=new_model, optimizer=new_optimizer, scheduler=new_scheduler
        )
        assert checkpoint["epoch"] == 2
        # Verify scheduler state was restored
        assert "step_size" in str(new_scheduler.state_dict())

    def test_load_checkpoint_returns_extra_keys(self, tmp_path: Path) -> None:
        """load_checkpoint should return extra keys in the checkpoint dict."""
        model = SimpleLinearModel()
        path = str(tmp_path / "checkpoint.pt")
        save_checkpoint(path, model=model, epoch=5, metrics={"accuracy": 0.95})
        checkpoint = load_checkpoint(path, model=SimpleLinearModel())
        assert checkpoint["epoch"] == 5
        assert checkpoint["metrics"]["accuracy"] == 0.95

    def test_save_checkpoint_without_optimizer(self, tmp_path: Path) -> None:
        """Checkpoint without optimizer should not have optimizer_state_dict."""
        model = SimpleLinearModel()
        path = str(tmp_path / "checkpoint.pt")
        save_checkpoint(path, model=model)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" not in checkpoint
        assert "scheduler_state_dict" not in checkpoint