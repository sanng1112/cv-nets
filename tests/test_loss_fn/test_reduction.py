from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn.reduction import reduce_loss


class TestReduceLoss:
    def test_reduce_mean(self):
        loss = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = reduce_loss(loss, reduction="mean")
        assert result.item() == pytest.approx(2.5)

    def test_reduce_sum(self):
        loss = torch.tensor([1.0, 2.0, 3.0])
        result = reduce_loss(loss, reduction="sum")
        assert result.item() == pytest.approx(6.0)

    def test_reduce_none(self):
        loss = torch.tensor([1.0, 2.0])
        result = reduce_loss(loss, reduction="none")
        assert result.shape == (2,)
        assert result[0].item() == pytest.approx(1.0)
        assert result[1].item() == pytest.approx(2.0)

    def test_reduce_mean_with_weight(self):
        loss = torch.tensor([1.0, 2.0, 3.0])
        weight = torch.tensor([0.1, 0.5, 0.4])
        result = reduce_loss(loss, reduction="mean", weight=weight)
        expected = (1.0 * 0.1 + 2.0 * 0.5 + 3.0 * 0.4) / (0.1 + 0.5 + 0.4)
        assert result.item() == pytest.approx(expected)
