"""
Tests for the Trainer class.

Covers creation, fitting with dummy data, validation, and device
assignment.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Tuple

import pytest
import torch
from torch import Tensor, nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset, TensorDataset

from cvnets.trainer import Trainer


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
    """Create random inputs and integer targets."""
    x = torch.randn(n_samples, in_features)
    y = torch.randint(0, num_classes, (n_samples,))
    return x, y


def _make_dataloader(
    x: Tensor, y: Tensor, batch_size: int = 4
) -> DataLoader[Tuple[Tensor, Tensor]]:
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainer:
    """Test suite for the ``Trainer`` class."""

    def test_trainer_creation(self) -> None:
        """Verify that a Trainer can be instantiated with default args."""
        model = SimpleLinearModel()
        loader = _make_dataloader(*_make_dummy_data())
        optimizer = SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = Trainer(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=2,
        )

        assert trainer.num_epochs == 2
        assert trainer.grad_accum_steps == 1
        assert hasattr(trainer, "device")
        assert hasattr(trainer, "metrics_tracker")
        assert hasattr(trainer, "_callback_list")
        assert trainer.should_stop is False
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0

    def test_trainer_fit(self) -> None:
        """Trainer.fit() should run without errors on dummy data."""
        model = SimpleLinearModel()
        loader = _make_dataloader(*_make_dummy_data())
        optimizer = SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = Trainer(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=2,
        )

        metrics = trainer.fit()
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "avg_loss" in metrics

    def test_trainer_with_validation(self) -> None:
        """Trainer with a validation loader should produce val_ metrics."""
        model = SimpleLinearModel()
        train_loader = _make_dataloader(*_make_dummy_data(16))
        val_loader = _make_dataloader(*_make_dummy_data(8))
        optimizer = SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            val_loader=val_loader,
            num_epochs=2,
        )

        metrics = trainer.fit()
        # Should have both training and validation metrics
        assert "val_accuracy" in metrics
        assert "val_avg_loss" in metrics

    def test_trainer_device(self) -> None:
        """Device should be correctly assigned."""
        model = SimpleLinearModel()
        loader = _make_dataloader(*_make_dummy_data())
        optimizer = SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Explicit CPU device
        trainer = Trainer(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
        )
        assert str(trainer.device) == "cpu"

        # Auto-detect should work (almost always CPU in CI)
        trainer2 = Trainer(
            model=SimpleLinearModel(),
            train_loader=loader,
            optimizer=optimizer,
            criterion=criterion,
        )
        assert isinstance(trainer2.device, torch.device)
