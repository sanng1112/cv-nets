"""
Integration tests that build ViT and CNN models via ``ModelFactory`` config.

Exercises the new ``multi_head_attention``, ``patch_embedding``, and existing
block-registry layer types end-to-end through the factory build pipeline.
"""

from __future__ import annotations

import torch

from cvnets.models.factory import ModelFactory


class TestViTIntegration:
    """End-to-end tests for building ViT-style and CNN-style models."""

    def test_build_minimal_vit_via_factory(self) -> None:
        """Build a minimal ViT: patch embedding → 2×Transformer → fc."""
        config = {
            "model": {
                "name": "TinyViT",
                "layers": [
                    {
                        "type": "patch_embedding",
                        "img_size": 32,
                        "patch_size": 8,
                        "in_channels": 3,
                        "embed_dim": 64,
                    },
                    {
                        "type": "TransformerEncoderBlock",
                        "embed_dim": 64,
                        "num_heads": 4,
                        "mlp_ratio": 2.0,
                        "dropout": 0.0,
                        "drop_path": 0.0,
                        "layer_scale_init": 0.0,
                    },
                    {
                        "type": "TransformerEncoderBlock",
                        "embed_dim": 64,
                        "num_heads": 4,
                        "mlp_ratio": 2.0,
                        "dropout": 0.0,
                        "drop_path": 0.0,
                        "layer_scale_init": 0.0,
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
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_build_cnn_with_se_block(self) -> None:
        """Build a CNN with a squeeze-and-excitation channel attention block."""
        config = {
            "model": {
                "name": "CNNwithSE",
                "layers": [
                    {
                        "type": "ConvBNAct",
                        "conv": {
                            "in_channels": 3,
                            "out_channels": 16,
                            "kernel_size": 3,
                        },
                        "act": {"type": "relu"},
                    },
                    {
                        "type": "SEBlock",
                        "in_channels": 16,
                        "reduction": 4,
                    },
                    {
                        "type": "fc",
                        "in_features": 16,
                        "out_features": 10,
                    },
                ],
            }
        }
        model = ModelFactory.build(config)
        x = torch.randn(2, 3, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)

    def test_build_cnn_with_depthwise_conv(self) -> None:
        """Build a CNN with depthwise separable convolutions."""
        config = {
            "model": {
                "name": "DWCNN",
                "layers": [
                    {
                        "type": "DepthwiseSeparableConvBlock",
                        "in_channels": 3,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 2,
                    },
                    {
                        "type": "DepthwiseSeparableConvBlock",
                        "in_channels": 32,
                        "out_channels": 64,
                        "kernel_size": 3,
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
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)
