"""
Tests for the model zoo — all registered architectures.

For each model:
  - Forward pass with correct input shape produces correct output shape
  - Gradient flows (backward succeeds)
  - Param count is reasonable
Also checks MODEL_REGISTRY contains all expected names.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")

import torch

from cvnets.core.registry import MODEL_REGISTRY

# Import to trigger registration
import cvnets.models.zoo  # noqa: F401


# ===================================================================
# Registry
# ===================================================================


class TestModelRegistry:
    """Verify all expected model names are registered."""

    EXPECTED_KEYS = {
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "mobilenet_v2",
        "vit_tiny",
        "vit_small",
        "vit_base",
    }

    def test_registry_contains_all_models(self) -> None:
        """All model names appear in MODEL_REGISTRY."""
        registered = set(MODEL_REGISTRY.keys())
        missing = self.EXPECTED_KEYS - registered
        assert not missing, f"Missing from registry: {missing}"

    def test_registry_build_resnet18(self) -> None:
        """Build resnet18 via registry and verify it's a module."""
        model = MODEL_REGISTRY.build("resnet18", num_classes=10)
        assert isinstance(model, torch.nn.Module)

    def test_registry_build_vit_base(self) -> None:
        """Build vit_base via registry."""
        model = MODEL_REGISTRY.build("vit_base", num_classes=1000)
        assert isinstance(model, torch.nn.Module)


# ===================================================================
# ResNet-18/34
# ===================================================================


class TestResNet18:
    """Tests for ResNet-18."""

    def test_forward_output_shape(self) -> None:
        """Output shape matches num_classes."""
        model = MODEL_REGISTRY.build("resnet18", num_classes=100)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 100), f"Expected (2, 100), got {out.shape}"

    def test_gradient_flows(self) -> None:
        """Backward pass produces gradients."""
        model = MODEL_REGISTRY.build("resnet18", num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_param_count_reasonable(self) -> None:
        """ResNet-18 should have ~11.2M parameters."""
        model = MODEL_REGISTRY.build("resnet18", num_classes=1000)
        n = sum(p.numel() for p in model.parameters())
        assert 10_000_000 < n < 13_000_000, f"Unexpected param count: {n}"


class TestResNet34:
    """Tests for ResNet-34."""

    def test_forward_output_shape(self) -> None:
        model = MODEL_REGISTRY.build("resnet34", num_classes=1000)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 1000)

    def test_gradient_flows(self) -> None:
        model = MODEL_REGISTRY.build("resnet34", num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_param_count_reasonable(self) -> None:
        """ResNet-34 should have ~21.8M parameters."""
        model = MODEL_REGISTRY.build("resnet34", num_classes=1000)
        n = sum(p.numel() for p in model.parameters())
        assert 20_000_000 < n < 23_000_000, f"Unexpected param count: {n}"


# ===================================================================
# ResNet-50/101
# ===================================================================


class TestResNet50:
    """Tests for ResNet-50."""

    def test_forward_output_shape(self) -> None:
        model = MODEL_REGISTRY.build("resnet50", num_classes=1000)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 1000)

    def test_gradient_flows(self) -> None:
        model = MODEL_REGISTRY.build("resnet50", num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_param_count_reasonable(self) -> None:
        """ResNet-50 should have ~25.6M parameters."""
        model = MODEL_REGISTRY.build("resnet50", num_classes=1000)
        n = sum(p.numel() for p in model.parameters())
        assert 24_000_000 < n < 27_000_000, f"Unexpected param count: {n}"


class TestResNet101:
    """Tests for ResNet-101."""

    def test_forward_output_shape(self) -> None:
        model = MODEL_REGISTRY.build("resnet101", num_classes=1000)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 1000)

    def test_gradient_flows(self) -> None:
        model = MODEL_REGISTRY.build("resnet101", num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_param_count_reasonable(self) -> None:
        """ResNet-101 should have ~44.5M parameters."""
        model = MODEL_REGISTRY.build("resnet101", num_classes=1000)
        n = sum(p.numel() for p in model.parameters())
        assert 43_000_000 < n < 46_000_000, f"Unexpected param count: {n}"


# ===================================================================
# MobileNetV2
# ===================================================================


class TestMobileNetV2:
    """Tests for MobileNetV2."""

    def test_forward_output_shape(self) -> None:
        model = MODEL_REGISTRY.build("mobilenet_v2", num_classes=100)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 100)

    def test_gradient_flows(self) -> None:
        model = MODEL_REGISTRY.build("mobilenet_v2", num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_param_count_reasonable(self) -> None:
        """MobileNetV2 should have ~3.5M parameters."""
        model = MODEL_REGISTRY.build("mobilenet_v2", num_classes=1000)
        n = sum(p.numel() for p in model.parameters())
        assert 3_000_000 < n < 4_000_000, f"Unexpected param count: {n}"


# ===================================================================
# ViT
# ===================================================================


class TestViTTiny:
    """Tests for ViT-Tiny."""

    def test_forward_output_shape(self) -> None:
        model = MODEL_REGISTRY.build("vit_tiny", num_classes=100)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 100)

    def test_gradient_flows(self) -> None:
        model = MODEL_REGISTRY.build("vit_tiny", num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_param_count_reasonable(self) -> None:
        """ViT-Tiny should have ~5.7M parameters."""
        model = MODEL_REGISTRY.build("vit_tiny", num_classes=1000)
        n = sum(p.numel() for p in model.parameters())
        assert 5_000_000 < n < 7_000_000, f"Unexpected param count: {n}"


class TestViTSmall:
    """Tests for ViT-Small."""

    def test_forward_output_shape(self) -> None:
        model = MODEL_REGISTRY.build("vit_small", num_classes=1000)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 1000)

    def test_gradient_flows(self) -> None:
        model = MODEL_REGISTRY.build("vit_small", num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_param_count_reasonable(self) -> None:
        """ViT-Small should have ~22M parameters."""
        model = MODEL_REGISTRY.build("vit_small", num_classes=1000)
        n = sum(p.numel() for p in model.parameters())
        assert 20_000_000 < n < 24_000_000, f"Unexpected param count: {n}"


class TestViTBase:
    """Tests for ViT-Base."""

    def test_forward_output_shape(self) -> None:
        model = MODEL_REGISTRY.build("vit_base", num_classes=1000)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 1000)

    def test_gradient_flows(self) -> None:
        model = MODEL_REGISTRY.build("vit_base", num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_param_count_reasonable(self) -> None:
        """ViT-Base should have ~86.5M parameters."""
        model = MODEL_REGISTRY.build("vit_base", num_classes=1000)
        n = sum(p.numel() for p in model.parameters())
        assert 80_000_000 < n < 90_000_000, f"Unexpected param count: {n}"


# ===================================================================
# Config consistency
# ===================================================================


class TestConfigCreatedArchitecture:
    """Verify that a direct call to factory functions creates correct architecture."""

    def test_resnet18_from_direct_call(self) -> None:
        """Calling make_resnet18 directly returns a valid model."""
        from cvnets.models.zoo.resnet import make_resnet18
        model = make_resnet18(num_classes=100)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 100)

    def test_mobilenet_v2_from_direct_call(self) -> None:
        """Calling make_mobilenet_v2 directly returns a valid model."""
        from cvnets.models.zoo.mobilenet_v2 import make_mobilenet_v2
        model = make_mobilenet_v2(num_classes=100)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 100)

    def test_vit_tiny_from_direct_call(self) -> None:
        """Calling make_vit_tiny directly returns a valid model."""
        from cvnets.models.zoo.vit import make_vit_tiny
        model = make_vit_tiny(patch_size=16, image_size=224, num_classes=100)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 100)
