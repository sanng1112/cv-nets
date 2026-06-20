"""
Tests for the scheduler package (registry, wrappers, factory).
"""
from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    StepLR,
)

from cvnets.scheduler import (
    SCHED_REGISTRY,
    build_scheduler,
    register_scheduler,
)
from cvnets.scheduler.cosine import CosineAnnealingLRWrapper
from cvnets.scheduler.step import StepLRWrapper
from cvnets.scheduler.one_cycle import OneCycleLRWrapper


class TestSchedulerRegistry:
    """Tests for SCHED_REGISTRY."""

    def test_registry_contains_cosine(self) -> None:
        assert SCHED_REGISTRY.contains("cosine")

    def test_registry_contains_step(self) -> None:
        assert SCHED_REGISTRY.contains("step")

    def test_registry_contains_one_cycle(self) -> None:
        assert SCHED_REGISTRY.contains("one_cycle")

    def test_registry_keys(self) -> None:
        keys = SCHED_REGISTRY.keys()
        assert "cosine" in keys
        assert "step" in keys
        assert "one_cycle" in keys

    def test_register_scheduler_decorator(self) -> None:
        @register_scheduler("test_dummy_sched")
        class DummyScheduler(nn.Module):
            def __init__(self, optimizer: Any, **kwargs: Any) -> None:
                super().__init__()
                self._sched = StepLR(optimizer, step_size=1)

            def step(self) -> None:
                self._sched.step()

        assert SCHED_REGISTRY.contains("test_dummy_sched")

    def test_register_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate registration"):

            @register_scheduler("cosine")
            class DuplicateCosine(nn.Module):  # type: ignore[no-redef]
                pass


class TestCosineAnnealingLRWrapper:
    """Tests for CosineAnnealingLRWrapper."""

    def test_create_cosine(self) -> None:
        model = nn.Linear(4, 2)
        optim = SGD(model.parameters(), lr=0.01)
        sched = CosineAnnealingLRWrapper(optim, T_max=10)
        assert isinstance(sched.scheduler, CosineAnnealingLR)
        assert sched.scheduler.T_max == 10

    def test_cosine_step_updates_lr(self) -> None:
        model = nn.Linear(4, 2)
        optim = SGD(model.parameters(), lr=0.1)
        sched = CosineAnnealingLRWrapper(optim, T_max=5)

        lrs = []
        for _ in range(5):
            lrs.append(optim.param_groups[0]["lr"])
            sched.step()
        # LR should decrease from 0.1 to eta_min (0.0)
        assert lrs[0] == 0.1
        assert lrs[-1] < lrs[0]

    def test_cosine_state_dict(self) -> None:
        model = nn.Linear(4, 2)
        optim = SGD(model.parameters(), lr=0.01)
        sched = CosineAnnealingLRWrapper(optim, T_max=5)
        sched.step()
        state = sched.state_dict()
        assert "last_epoch" in state

        sched2 = CosineAnnealingLRWrapper(
            SGD(nn.Linear(4, 2).parameters(), lr=0.01), T_max=5
        )
        sched2.load_state_dict(state)
        assert sched2.scheduler.last_epoch == sched.scheduler.last_epoch


class TestStepLRWrapper:
    """Tests for StepLRWrapper."""

    def test_create_step(self) -> None:
        model = nn.Linear(4, 2)
        optim = SGD(model.parameters(), lr=0.01)
        sched = StepLRWrapper(optim, step_size=3, gamma=0.1)
        assert isinstance(sched.scheduler, StepLR)
        assert sched.scheduler.step_size == 3

    def test_step_lr_decay(self) -> None:
        model = nn.Linear(4, 2)
        optim = SGD(model.parameters(), lr=0.1)
        sched = StepLRWrapper(optim, step_size=2, gamma=0.1)

        lrs = []
        for _ in range(4):
            lrs.append(optim.param_groups[0]["lr"])
            sched.step()
        assert lrs[0] == pytest.approx(0.1, rel=1e-5)   # epoch 1
        assert lrs[1] == pytest.approx(0.1, rel=1e-5)   # epoch 2 (before step)
        assert lrs[2] == pytest.approx(0.01, rel=1e-5)  # epoch 3 (after first decay)
        assert lrs[3] == pytest.approx(0.01, rel=1e-5)  # epoch 4 (before second decay)


class TestOneCycleLRWrapper:
    """Tests for OneCycleLRWrapper."""

    def test_create_one_cycle(self) -> None:
        model = nn.Linear(4, 2)
        optim = SGD(model.parameters(), lr=0.01)
        sched = OneCycleLRWrapper(optim, max_lr=0.1, total_steps=10)
        assert isinstance(sched.scheduler, OneCycleLR)

    def test_one_cycle_lr_changes(self) -> None:
        model = nn.Linear(4, 2)
        optim = SGD(model.parameters(), lr=0.01)
        sched = OneCycleLRWrapper(optim, max_lr=0.1, total_steps=10)

        first_lr = optim.param_groups[0]["lr"]
        for _ in range(5):
            sched.step()
        mid_lr = optim.param_groups[0]["lr"]
        for _ in range(4):
            sched.step()
        last_lr = optim.param_groups[0]["lr"]
        # OneCycleLR should increase then decrease
        assert last_lr < mid_lr  # last should be lower than mid


class TestBuildScheduler:
    """Tests for build_scheduler factory."""

    def _make_optim(self) -> SGD:
        model = nn.Linear(4, 2)
        return SGD(model.parameters(), lr=0.01)

    def test_build_cosine(self) -> None:
        sched = build_scheduler(self._make_optim(), "cosine", T_max=10)
        assert isinstance(sched, CosineAnnealingLRWrapper)

    def test_build_step(self) -> None:
        sched = build_scheduler(self._make_optim(), "step", step_size=3)
        assert isinstance(sched, StepLRWrapper)

    def test_build_one_cycle(self) -> None:
        sched = build_scheduler(
            self._make_optim(), "one_cycle", max_lr=0.1, total_steps=10
        )
        assert isinstance(sched, OneCycleLRWrapper)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown scheduler"):
            build_scheduler(self._make_optim(), "unknown_sched")

    def test_build_cosine_with_kwargs(self) -> None:
        sched = build_scheduler(
            self._make_optim(), "cosine", T_max=20, eta_min=1e-6
        )
        assert sched.scheduler.T_max == 20
        assert sched.scheduler.eta_min == 1e-6

    def test_build_step_with_kwargs(self) -> None:
        sched = build_scheduler(
            self._make_optim(), "step", step_size=5, gamma=0.5
        )
        assert sched.scheduler.step_size == 5
        assert sched.scheduler.gamma == 0.5