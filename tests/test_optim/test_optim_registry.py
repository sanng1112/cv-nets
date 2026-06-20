"""
Tests for the optimizer registry (OPTIM_REGISTRY, register_optimizer).
"""
from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from cvnets.optim.registry import OPTIM_REGISTRY, register_optimizer


class TestOptimRegistry:
    """Test suite for OPTIM_REGISTRY and register_optimizer."""

    def test_registry_contains_sgd(self) -> None:
        """OPTIM_REGISTRY should contain 'sgd'."""
        assert OPTIM_REGISTRY.contains("sgd")

    def test_registry_contains_adam(self) -> None:
        """OPTIM_REGISTRY should contain 'adam'."""
        assert OPTIM_REGISTRY.contains("adam")

    def test_registry_contains_adamw(self) -> None:
        """OPTIM_REGISTRY should contain 'adamw'."""
        assert OPTIM_REGISTRY.contains("adamw")

    def test_registry_keys(self) -> None:
        """Registered optimizer keys should include sgd, adam, adamw."""
        keys = OPTIM_REGISTRY.keys()
        assert "sgd" in keys
        assert "adam" in keys
        assert "adamw" in keys

    def test_register_optimizer_decorator(self) -> None:
        """register_optimizer decorator registers in OPTIM_REGISTRY."""

        @register_optimizer("test_dummy_optim")
        class DummyOptimWrapper(nn.Module):
            def __init__(self, params: Any, lr: float = 0.01) -> None:
                super().__init__()
                self._optimizer = torch.optim.SGD(params, lr=lr)

            @property
            def optimizer(self) -> torch.optim.SGD:
                return self._optimizer

        assert OPTIM_REGISTRY.contains("test_dummy_optim")
        model = nn.Linear(4, 2)
        instance = OPTIM_REGISTRY.build(
            "test_dummy_optim", params=model.parameters(), lr=0.1
        )
        assert isinstance(instance, nn.Module)
        assert hasattr(instance, "optimizer")

    def test_register_duplicate_raises(self) -> None:
        """Duplicate registration should raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate registration"):

            @register_optimizer("sgd")
            class DuplicateSGD(nn.Module):  # type: ignore[no-redef]
                pass

    def test_build_sgd_returns_sgdwrapper(self) -> None:
        """Building 'sgd' should return an SGDWrapper instance."""
        from cvnets.optim.sgd import SGDWrapper

        model = nn.Linear(4, 2)
        instance = OPTIM_REGISTRY.build("sgd", params=model.parameters(), lr=0.01)
        assert isinstance(instance, SGDWrapper)

    def test_build_adam_returns_adamwrapper(self) -> None:
        """Building 'adam' should return an AdamWrapper instance."""
        from cvnets.optim.adam import AdamWrapper

        model = nn.Linear(4, 2)
        instance = OPTIM_REGISTRY.build("adam", params=model.parameters(), lr=0.001)
        assert isinstance(instance, AdamWrapper)

    def test_build_adamw_returns_adamwwrapper(self) -> None:
        """Building 'adamw' should return an AdamWWrapper instance."""
        from cvnets.optim.adam import AdamWWrapper

        model = nn.Linear(4, 2)
        instance = OPTIM_REGISTRY.build("adamw", params=model.parameters(), lr=0.001)
        assert isinstance(instance, AdamWWrapper)

    def test_build_missing_key_raises(self) -> None:
        """Building a missing key should raise KeyError."""
        with pytest.raises(KeyError, match="missing_optim"):
            OPTIM_REGISTRY.build("missing_optim")
