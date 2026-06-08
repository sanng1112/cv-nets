"""Tests for the DropPath (Stochastic Depth) layer."""

from __future__ import annotations

import torch
import pytest
from cvnets.layers.drop_path import DropPath


class TestDropPath:
    """Test suite for DropPath."""

    def test_train_mode_drops_some(self) -> None:
        dp = DropPath(drop_prob=0.5)
        dp.train()
        torch.manual_seed(42)
        x = torch.ones(1000, 4, 16)
        out = dp(x)
        per_item_sum = out.abs().sum(dim=(1, 2))
        num_zeroed = (per_item_sum < 1e-6).sum().item()
        assert 200 < num_zeroed < 800

    def test_eval_mode_no_drop(self) -> None:
        dp = DropPath(drop_prob=0.9)
        dp.eval()
        x = torch.randn(8, 16)
        out = dp(x)
        assert torch.allclose(out, x)

    def test_drop_prob_zero_is_identity(self) -> None:
        dp = DropPath(drop_prob=0.0)
        dp.train()
        x = torch.randn(8, 16)
        out = dp(x)
        assert torch.allclose(out, x)

    def test_drop_prob_one_drops_all_in_train(self) -> None:
        dp = DropPath(drop_prob=1.0)
        dp.train()
        x = torch.randn(8, 16)
        out = dp(x)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_survival_scaling(self) -> None:
        dp = DropPath(drop_prob=0.3)
        dp.train()
        torch.manual_seed(123)
        x = torch.ones(100, 1, 1)
        out = dp(x)
        survivors = out[out.abs() > 1e-6]
        assert len(survivors) > 0
        expected_scale = 1.0 / (1.0 - 0.3)
        assert torch.allclose(survivors, torch.full_like(survivors, expected_scale))
