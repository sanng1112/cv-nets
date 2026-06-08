from __future__ import annotations

import torch
import pytest
from cvnets.loss_fn import SUPPORTED_LOSSES, build_loss_fn, register_loss_fn
from cvnets.loss_fn.base_loss import BaseLoss


class TestLossRegistration:

    def test_registry_contains_cross_entropy(self):
        from cvnets.core.registry import LOSS_REGISTRY
        assert LOSS_REGISTRY.contains("cross_entropy", category="classification")

    def test_build_cross_entropy(self):
        loss_fn = build_loss_fn("cross_entropy", category="classification")
        assert isinstance(loss_fn, BaseLoss)
        p = torch.randn(4, 10)
        t = torch.randint(0, 10, (4,))
        out = loss_fn(p, t)
        assert out.shape == ()

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            build_loss_fn("nonexistent_loss", category="classification")

    def test_supported_losses_contains_classification(self):
        keys = [k for k in SUPPORTED_LOSSES if k.startswith("classification/")]
        assert len(keys) >= 5  # cross_entropy, focal_loss, asymmetric_loss, arcface_loss, cosface_loss

    def test_supported_losses_contains_segmentation(self):
        keys = [k for k in SUPPORTED_LOSSES if k.startswith("segmentation/")]
        assert len(keys) >= 4

    def test_supported_losses_contains_detection(self):
        keys = [k for k in SUPPORTED_LOSSES if k.startswith("detection/")]
        assert len(keys) >= 2

    def test_supported_losses_contains_metric_learning(self):
        keys = [k for k in SUPPORTED_LOSSES if k.startswith("metric_learning/")]
        assert len(keys) >= 4

    def test_supported_losses_contains_ssl(self):
        keys = [k for k in SUPPORTED_LOSSES if k.startswith("ssl/")]
        assert len(keys) >= 3

    def test_supported_losses_contains_regression(self):
        keys = [k for k in SUPPORTED_LOSSES if k.startswith("regression/")]
        assert len(keys) >= 3

    def test_register_custom_loss(self):
        class MyLoss(BaseLoss):
            def forward(self, prediction, target):
                return torch.tensor(0.0)

        register_loss_fn("my_custom_loss", category="test")(MyLoss)
        fn = build_loss_fn("my_custom_loss", category="test")
        assert isinstance(fn, MyLoss)

    def test_build_focal_loss(self):
        loss_fn = build_loss_fn("focal_loss", category="classification", gamma=2.0)
        assert isinstance(loss_fn, BaseLoss)
        p = torch.randn(4, 10)
        t = torch.randint(0, 10, (4,))
        out = loss_fn(p, t)
        assert out.shape == ()
