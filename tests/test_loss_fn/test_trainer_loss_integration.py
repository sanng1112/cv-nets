from __future__ import annotations

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from cvnets.loss_fn import build_loss_fn
from cvnets.trainer.trainer import Trainer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class TestTrainerLossIntegration:

    def test_trainer_accepts_cross_entropy(self):
        model = SimpleModel()
        data = torch.randn(20, 10)
        labels = torch.randint(0, 5, (20,))
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=4)
        criterion = build_loss_fn("cross_entropy", category="classification")
        optimizer = SGD(model.parameters(), lr=0.01)

        trainer = Trainer(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=1,
            device="cpu",
        )
        metrics = trainer.fit()
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "avg_loss" in metrics
        assert 0 <= metrics["accuracy"] <= 100
        assert metrics["avg_loss"] >= 0

    def test_trainer_with_focal_loss(self):
        model = SimpleModel()
        data = torch.randn(20, 10)
        labels = torch.randint(0, 5, (20,))
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=4)
        criterion = build_loss_fn("focal_loss", category="classification", gamma=2.0)
        optimizer = SGD(model.parameters(), lr=0.01)

        trainer = Trainer(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=1,
            device="cpu",
        )
        metrics = trainer.fit()
        assert "avg_loss" in metrics
        assert metrics["avg_loss"] >= 0
