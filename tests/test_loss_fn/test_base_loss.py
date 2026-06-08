from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn.base_loss import BaseLoss


class TestBaseLoss:
    def test_base_loss_abstract(self):
        """Cannot instantiate BaseLoss directly."""
        with pytest.raises(TypeError):
            BaseLoss()  # type: ignore

    def test_base_loss_concrete(self):
        """Subclass must implement forward."""

        class ConcreteLoss(BaseLoss):
            def forward(self, prediction, target):
                return torch.tensor(0.0)

        loss_fn = ConcreteLoss(reduction="mean")
        p = torch.randn(4, 10)
        t = torch.randint(0, 10, (4,))
        out = loss_fn(p, t)
        assert out.shape == ()

    def test_base_loss_extra_repr(self):
        """extra_repr includes reduction."""

        class ConcreteLoss(BaseLoss):
            def forward(self, prediction, target):
                return torch.tensor(0.0)

        loss_fn = ConcreteLoss(reduction="sum")
        r = loss_fn.extra_repr()
        assert "reduction" in r
        assert "sum" in r
