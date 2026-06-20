"""Tests for cvnets.research.probe.LayerProbe."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from cvnets.research.probe import LayerProbe


class SimpleModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestLayerProbe:

    @pytest.fixture
    def module(self) -> SimpleModule:
        return SimpleModule()

    def test_attach_detach(self, module: SimpleModule) -> None:
        probe = LayerProbe()
        probe.attach(module.linear)
        x = torch.randn(3, 4)
        out = module(x)
        out.sum().backward()
        assert len(probe.activations) == 1
        assert len(probe.gradients) == 1
        probe.detach()
        probe.clear()
        x2 = torch.randn(3, 4)
        out2 = module(x2)
        (out2.sum()).backward()
        assert len(probe.activations) == 0
        assert len(probe.gradients) == 0

    def test_activations_shape(self, module: SimpleModule) -> None:
        probe = LayerProbe()
        probe.attach(module.linear)
        module(torch.randn(3, 4))
        assert probe.activations[0].shape == (3, 2)

    def test_gradients_shape(self, module: SimpleModule) -> None:
        probe = LayerProbe()
        probe.attach(module.linear)
        x = torch.randn(3, 4)
        out = module(x)
        out.sum().backward()
        assert probe.gradients[0].shape == (3, 2)

    def test_clear(self, module: SimpleModule) -> None:
        probe = LayerProbe()
        probe.attach(module.linear)
        module(torch.randn(3, 4))
        assert len(probe.activations) == 1
        probe.clear()
        assert len(probe.activations) == 0

    def test_multiple_forward_passes(self, module: SimpleModule) -> None:
        probe = LayerProbe()
        probe.attach(module.linear)
        for _ in range(5):
            module(torch.randn(3, 4))
        assert len(probe.activations) == 5

    def test_attach_multiple_layers(self) -> None:
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        probe = LayerProbe()
        probe.attach(model[0])
        probe.attach(model[2])
        model(torch.randn(3, 4))
        assert len(probe.activations) == 2

    def test_context_manager(self) -> None:
        module = nn.Linear(4, 2)
        with LayerProbe() as probe:
            probe.attach(module)
            module(torch.randn(3, 4))
            assert len(probe.activations) == 1
        probe.clear()
        module(torch.randn(3, 4))
        assert len(probe.activations) == 0

    def test_detach_all(self, module: SimpleModule) -> None:
        probe = LayerProbe()
        probe.attach(module.linear)
        module(torch.randn(3, 4))
        assert len(probe.activations) == 1
        probe.detach_all()
        probe.clear()
        module(torch.randn(3, 4))
        assert len(probe.activations) == 0
