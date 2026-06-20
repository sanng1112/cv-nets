"""Tests for misc utilities."""

import torch
from torch import nn

from cvnets.utils import count_parameters, set_seed


class TestMisc:
    def test_count_parameters(self):
        model = nn.Linear(10, 5)
        total, trainable = count_parameters(model)
        assert total == 10 * 5 + 5  # weights + bias
        assert trainable == total

    def test_count_parameters_frozen(self):
        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad = False
        total, trainable = count_parameters(model)
        assert trainable == 0

    def test_set_seed_reproducible(self):
        set_seed(42)
        a = torch.randn(3, 3)
        set_seed(42)
        b = torch.randn(3, 3)
        assert (a == b).all()

    def test_set_seed_deterministic(self):
        """deterministic=True should not raise."""
        set_seed(42, deterministic=True)
        assert True
