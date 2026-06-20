"""
Tests for ``ModelFactory`` and the model build pipeline.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")

import unittest

import torch

from cvnets.models.factory import ModelFactory


class TestModelFactory(unittest.TestCase):
    """Integration tests for ``ModelFactory.build``."""

    def test_build_simple_cnn(self):
        """Build a minimal CNN with ConvBNAct + fc layer and verify forward."""
        config = {
            "model": {
                "name": "test_cnn",
                "layers": [
                    {
                        "type": "ConvBNAct",
                        "conv": {
                            "in_channels": 1,
                            "out_channels": 4,
                            "kernel_size": 3,
                        },
                        "act": {"type": "relu"},
                    },
                    {
                        "type": "fc",
                        "in_features": 4,
                        "out_features": 10,
                    },
                ],
            }
        }
        model = ModelFactory.build(config)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_build_empty_raises(self):
        """Building from an empty config should raise an exception."""
        with self.assertRaises(Exception):
            ModelFactory.build({})

    def test_build_unknown_layer_raises(self):
        """An unrecognised layer type should raise ``ValueError``."""
        config = {
            "model": {
                "name": "bad_model",
                "layers": [
                    {"type": "nonexistent_super_layer"},
                ],
            }
        }
        with self.assertRaises(ValueError):
            ModelFactory.build(config)

    def test_build_with_activation_layer(self):
        """An ``act`` type should produce an activation module."""
        config = {
            "model": {
                "name": "act_test",
                "layers": [
                    {
                        "type": "act",
                        "act_type": "relu",
                    },
                ],
            }
        }
        model = ModelFactory.build(config)
        x = torch.randn(2, 16)
        out = model(x)
        self.assertEqual(out.shape, (2, 16))

    def test_build_with_pooling(self):
        """Pooling layers should be accepted in feature extractor."""
        config = {
            "model": {
                "name": "pool_test",
                "layers": [
                    {
                        "type": "avgpool",
                        "kernel_size": 2,
                        "stride": 2,
                    },
                    {
                        "type": "fc",
                        "in_features": 64,
                        "out_features": 10,
                    },
                ],
            }
        }
        model = ModelFactory.build(config)
        x = torch.randn(2, 64, 4, 4)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))


if __name__ == "__main__":
    unittest.main()
