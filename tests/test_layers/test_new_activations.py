"""
Tests for the newly added activation functions: GELU, SiLU, Mish,
LeakyReLU, PReLU, ELU, Dropout, Dropout2d.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
from torch import nn

from cvnets.layers.activation import (
    SUPPORTED_ACT_FNS,
    ACT_FN_MODULES,
    build_activation_layer,
)


class TestNewActivations:
    """Test suite for the new activation functions."""

    ACTIVATION_CASES: Dict[str, Dict[str, Any]] = {
        "gelu": {"forward_test": {}, "build_kwargs": {}},
        "silu": {"forward_test": {}, "build_kwargs": {}},
        "mish": {"forward_test": {}, "build_kwargs": {}},
        "leaky_relu": {
            "forward_test": {"negative_slope": 0.01, "inplace": False},
            "build_kwargs": {"negative_slope": 0.1},
        },
        "prelu": {
            "forward_test": {"num_parameters": 1},
            "build_kwargs": {"num_parameters": 1},
        },
        "elu": {
            "forward_test": {"alpha": 1.0, "inplace": False},
            "build_kwargs": {"alpha": 1.0},
        },
        "dropout": {
            "forward_test": {"p": 0.0},
            "build_kwargs": {"p": 0.5},
        },
        "dropout2d": {
            "forward_test": {"p": 0.0},
            "build_kwargs": {"p": 0.5},
        },
    }

    @pytest.fixture(autouse=True)
    def _resolve_classes(self) -> None:
        for name, info in self.ACTIVATION_CASES.items():
            info["cls"] = ACT_FN_MODULES.get(name)

    def test_all_are_registered(self) -> None:
        """Check that all expected activations are in the registry."""
        expected = {
            "gelu", "silu", "mish", "leaky_relu", "prelu", "elu",
            "dropout", "dropout2d",
        }
        missing = expected - set(SUPPORTED_ACT_FNS)
        assert not missing, f"Missing activations: {missing}"

    def test_forward_shape(self) -> None:
        """All activations preserve the input shape."""
        x = torch.randn(2, 16, 32, 32)
        for name, info in self.ACTIVATION_CASES.items():
            if info["cls"] is None:
                pytest.skip(f"{name} not registered")
            act = info["cls"](**info["forward_test"])
            if "dropout" in name:
                act.eval()
            out = act(x)
            assert out.shape == x.shape, f"{name} changed shape"

    def test_build_via_factory(self) -> None:
        """Verify that ``build_activation_layer`` can construct each."""
        for name, info in self.ACTIVATION_CASES.items():
            if info["cls"] is None:
                pytest.skip(f"{name} not registered")
            kwargs = info.get("build_kwargs", {}).copy()
            kwargs["type"] = name
            act = build_activation_layer(opts=kwargs, act_type=name)
            assert act is not None, f"build_activation_layer returned None for {name}"
            assert isinstance(act, info["cls"]), (
                f"Expected {info['cls']}, got {type(act)} for {name}"
            )

    def test_mish_compute(self) -> None:
        """Mish: verify against known numerical values."""
        if "mish" not in ACT_FN_MODULES:
            pytest.skip("Mish not registered")
        mish = ACT_FN_MODULES["mish"]()
        x = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0])
        out = mish(x)
        assert torch.allclose(out[0], torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(out[1], torch.tensor(0.865098), atol=1e-4)
        assert torch.allclose(out[2], torch.tensor(-0.303401), atol=1e-4)

    def test_leaky_relu_negative_slope(self) -> None:
        """LeakyReLU: verify negative slope behavior."""
        if "leaky_relu" not in ACT_FN_MODULES:
            pytest.skip("LeakyReLU not registered")
        lr = ACT_FN_MODULES["leaky_relu"](negative_slope=0.1)
        x = torch.tensor([-1.0, 0.0, 1.0])
        out = lr(x)
        assert torch.allclose(out[0], torch.tensor(-0.1)), "negative_slope failed"
        assert torch.allclose(out[1], torch.tensor(0.0))
        assert torch.allclose(out[2], torch.tensor(1.0))

    def test_prelu_learnable(self) -> None:
        """PReLU: has a learnable parameter."""
        if "prelu" not in ACT_FN_MODULES:
            pytest.skip("PReLU not registered")
        p = ACT_FN_MODULES["prelu"](num_parameters=1)
        assert hasattr(p, "weight")
        assert p.weight.numel() == 1
        assert p.weight.requires_grad

    def test_dropout_eval_identity(self) -> None:
        """Dropout and Dropout2d are identity in eval mode."""
        x = torch.randn(4, 8, 16, 16)
        for name in ("dropout", "dropout2d"):
            if name not in ACT_FN_MODULES:
                pytest.skip(f"{name} not registered")
            d = ACT_FN_MODULES[name](p=0.5)
            d.eval()
            out = d(x)
            assert torch.allclose(out, x), f"{name} not identity in eval"

    def test_gradient_flows(self) -> None:
        """Check that backward pass works for all activations."""
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        for name, info in self.ACTIVATION_CASES.items():
            if info["cls"] is None:
                pytest.skip(f"{name} not registered")
            act = info["cls"](**info["forward_test"])
            if "dropout" in name:
                act.eval()
            out = act(x)
            loss = out.sum()
            loss.backward(retain_graph=True)
            assert x.grad is not None, f"{name} has no gradient"
            assert not torch.allclose(
                x.grad, torch.zeros_like(x.grad)
            ), f"{name} gradient is zero"
            x.grad.zero_()
