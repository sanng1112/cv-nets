"""Tests for transform pipeline."""

import pytest
import torch
from torch import Tensor

from cvnets.data.transforms import (
    build_transform_pipeline,
    SUPPORTED_TRANSFORMS,
)


class TestTransformPipeline:
    def test_build_transform_pipeline_returns_compose(self):
        """build_transform_pipeline with a list config returns a callable."""
        config = [
            {"type": "resize", "size": 32},
            {"type": "to_tensor"},
        ]
        pipeline = build_transform_pipeline(config)
        assert callable(pipeline)

    def test_empty_config_returns_identity(self):
        """Empty or None config returns identity (passthrough)."""
        identity = build_transform_pipeline([])
        x = torch.randn(3, 32, 32)
        result = identity(x)
        assert isinstance(result, Tensor)
        assert result.shape == (3, 32, 32)

    def test_resize_transform(self):
        """'resize' transform changes spatial dims."""
        pipeline = build_transform_pipeline([{"type": "resize", "size": 16}])
        x = torch.randn(3, 32, 32)
        result = pipeline(x)
        assert result.shape[-2:] == (16, 16)

    def test_to_tensor_transform(self):
        """'to_tensor' converts ndarray to float tensor."""
        import numpy as np
        pipeline = build_transform_pipeline([{"type": "to_tensor"}])
        x = np.random.randn(32, 32, 3).astype(np.float32)
        result = pipeline(x)
        assert isinstance(result, Tensor)
        assert result.shape == (3, 32, 32)
        assert result.dtype == torch.float32

    def test_unknown_transform_raises(self):
        """Unknown transform type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown transform"):
            build_transform_pipeline([{"type": "nonexistent_transform"}])

    def test_supported_transforms_listed(self):
        """SUPPORTED_TRANSFORMS contains expected keys."""
        assert "resize" in SUPPORTED_TRANSFORMS
        assert "to_tensor" in SUPPORTED_TRANSFORMS
        assert "normalize" in SUPPORTED_TRANSFORMS
        assert "random_horizontal_flip" in SUPPORTED_TRANSFORMS
