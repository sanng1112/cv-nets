"""
Tests for the build_optimizer factory function.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW

from cvnets.optim import build_optimizer


class TestBuildOptimizer:
    """Test suite for build_optimizer factory."""

    def test_build_sgd_defaults(self) -> None:
        """Build SGD with default parameters."""
        model = nn.Linear(4, 2)
        optim = build_optimizer(model.parameters(), "sgd", lr=0.01)
        assert hasattr(optim, "optimizer")
        assert isinstance(optim.optimizer, SGD)
        assert optim.optimizer.param_groups[0]["lr"] == 0.01

    def test_build_sgd_with_momentum(self) -> None:
        """Build SGD with momentum."""
        model = nn.Linear(4, 2)
        optim = build_optimizer(model.parameters(), "sgd", lr=0.01, momentum=0.9)
        assert optim.optimizer.param_groups[0]["momentum"] == 0.9

    def test_build_sgd_with_weight_decay(self) -> None:
        """Build SGD with weight decay."""
        model = nn.Linear(4, 2)
        optim = build_optimizer(
            model.parameters(), "sgd", lr=0.01, weight_decay=1e-4
        )
        assert optim.optimizer.param_groups[0]["weight_decay"] == 1e-4

    def test_build_sgd_nesterov(self) -> None:
        """Build SGD with Nesterov momentum."""
        model = nn.Linear(4, 2)
        optim = build_optimizer(
            model.parameters(), "sgd", lr=0.01, momentum=0.9, nesterov=True
        )
        assert optim.optimizer.param_groups[0]["nesterov"] is True

    def test_build_adam_defaults(self) -> None:
        """Build Adam with default parameters."""
        model = nn.Linear(4, 2)
        optim = build_optimizer(model.parameters(), "adam", lr=0.001)
        assert isinstance(optim.optimizer, Adam)
        assert optim.optimizer.param_groups[0]["lr"] == 0.001
        assert optim.optimizer.param_groups[0]["betas"] == (0.9, 0.999)

    def test_build_adam_custom_betas(self) -> None:
        """Build Adam with custom betas."""
        model = nn.Linear(4, 2)
        optim = build_optimizer(
            model.parameters(), "adam", lr=0.001, betas=(0.8, 0.99)
        )
        assert optim.optimizer.param_groups[0]["betas"] == (0.8, 0.99)

    def test_build_adamw_defaults(self) -> None:
        """Build AdamW with default parameters."""
        model = nn.Linear(4, 2)
        optim = build_optimizer(model.parameters(), "adamw", lr=0.001)
        assert isinstance(optim.optimizer, AdamW)

    def test_build_unknown_raises(self) -> None:
        """Build with unknown type should raise ValueError."""
        model = nn.Linear(4, 2)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer(model.parameters(), "unknown_optim")

    def test_build_verbose(self, capsys: Any) -> None:
        """Verbose mode should print info about the optimizer."""
        model = nn.Linear(4, 2)
        build_optimizer(model.parameters(), "sgd", lr=0.01, verbose=True)
        captured = capsys.readouterr()
        assert "Building optimizer: sgd" in captured.out

    def test_build_sgd_optimizer_step(self) -> None:
        """SGDWrapper should support step() and zero_grad()."""
        model = nn.Linear(4, 2)
        optim = build_optimizer(model.parameters(), "sgd", lr=0.01)

        x = torch.randn(2, 4)
        y = torch.randn(2, 2)
        loss = model(x).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()
        # Just verify no exception

    def test_build_adam_state_dict(self) -> None:
        """AdamWrapper should support state_dict and load_state_dict."""
        model = nn.Linear(4, 2)
        optim = build_optimizer(model.parameters(), "adam", lr=0.001)

        x = torch.randn(2, 4)
        y = torch.randn(2, 2)
        loss = model(x).sum()
        loss.backward()
        optim.step()

        state = optim.state_dict()
        assert "state" in state
        assert "param_groups" in state

        # Re-load state dict
        model2 = nn.Linear(4, 2)
        optim2 = build_optimizer(model2.parameters(), "adam", lr=0.001)
        optim2.load_state_dict(state)
        # Verify the state was restored
        assert optim2.state_dict()["param_groups"] == state["param_groups"]

    def test_build_sgd_param_groups_property(self) -> None:
        """SGDWrapper should expose param_groups."""
        model = nn.Linear(4, 2)
        optim = build_optimizer(model.parameters(), "sgd", lr=0.01)
        param_groups = optim.param_groups
        assert isinstance(param_groups, list)
        assert len(param_groups) > 0
        assert "lr" in param_groups[0]

    def test_build_with_parameter_groups(self) -> None:
        """Build optimizer with parameter groups (dict-based params)."""
        model = nn.Linear(4, 2)
        params = [
            {"params": model.weight, "lr": 0.01},
            {"params": model.bias, "lr": 0.001},
        ]
        optim = build_optimizer(params, "sgd", lr=0.01)
        assert len(optim.param_groups) == 2
        assert optim.param_groups[0]["lr"] == 0.01
        assert optim.param_groups[1]["lr"] == 0.001
