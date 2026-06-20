"""
Simple CNN — a minimal convolutional model built via ``ModelFactory``.

Provides a single convenience function ``simple_cnn()`` that returns a
``_ComposedModel`` instance suitable for small-scale experiments (e.g.
MNIST).
"""

from __future__ import annotations

from cvnets.models.factory import ModelFactory


def simple_cnn(
    in_channels: int = 1,
    hidden_channels: int = 32,
    num_classes: int = 10,
    image_size: int = 28,
) -> "ModelFactory._ComposedModel":  # noqa: F821
    """Create a simple 3-conv + pooling + FC CNN.

    The feature extractor consists of three ``ConvBNAct`` blocks (each
    doubling the channel count) followed by an average pooling layer that
    reduces spatial dimensions to 1×1.  The classifier is a single
    fully-connected layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 1 for grayscale).
    hidden_channels : int
        Base channel count; each subsequent conv doubles it.
    num_classes : int
        Number of output classes.
    image_size : int
        Height (and width) of the input image.

    Returns
    -------
    _ComposedModel
        The constructed model.
    """
    config = {
        "model": {
            "name": "SimpleCNN",
            "layers": [
                {
                    "type": "ConvBNAct",
                    "conv": {
                        "in_channels": in_channels,
                        "out_channels": hidden_channels,
                        "kernel_size": 3,
                    },
                    "act": {"type": "relu"},
                },
                {
                    "type": "ConvBNAct",
                    "conv": {
                        "in_channels": hidden_channels,
                        "out_channels": hidden_channels * 2,
                        "kernel_size": 3,
                    },
                    "act": {"type": "relu"},
                },
                {
                    "type": "ConvBNAct",
                    "conv": {
                        "in_channels": hidden_channels * 2,
                        "out_channels": hidden_channels * 4,
                        "kernel_size": 3,
                    },
                    "act": {"type": "relu"},
                },
                {
                    "type": "avgpool",
                    "kernel_size": image_size // 4,
                    "stride": 1,
                },
                {
                    "type": "fc",
                    "in_features": hidden_channels * 4,
                    "out_features": num_classes,
                },
            ],
        }
    }
    return ModelFactory.build(config)
