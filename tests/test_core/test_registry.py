"""
Tests for the ``Registry`` class and global singletons.
"""

from __future__ import annotations

from typing import Any

import pytest

from cvnets.core.registry import (
    ACTIVATION_REGISTRY,
    BLOCK_REGISTRY,
    LOSS_REGISTRY,
    MODEL_REGISTRY,
    NORMALIZATION_REGISTRY,
    POOLING_REGISTRY,
    Registry,
)


# ===================================================================
# Helper Dummy classes
# ===================================================================

class DummyModel:
    """Minimal dummy used for registration tests."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class AnotherModel:
    """Second dummy for category-based tests."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


# ===================================================================
# TestRegistry
# ===================================================================

class TestRegistry:
    """Test suite for Registry."""

    def test_register_and_build(self) -> None:
        """Register a Dummy class, build it, verify instance."""
        reg = Registry("test_basic")

        @reg.register("dummy")
        class _(DummyModel):
            pass

        instance = reg.build("dummy", x=1, y=2)
        assert isinstance(instance, DummyModel)
        assert instance.x == 1
        assert instance.y == 2

    def test_register_duplicate_raises(self) -> None:
        """Duplicate registration raises ValueError."""
        reg = Registry("test_dup")

        @reg.register("dup_key")
        class First:
            pass

        with pytest.raises(ValueError, match="Duplicate registration"):
            @reg.register("dup_key")
            class Second:  # type: ignore[no-redef]
                pass

    def test_build_missing_key_raises(self) -> None:
        """Missing key raises KeyError with helpful message."""
        reg = Registry("test_missing")

        @reg.register("existing")
        class Existing:
            pass

        with pytest.raises(KeyError, match="missing_key"):
            reg.build("missing_key")

        # Also raises when category is wrong
        with pytest.raises(KeyError, match="missing_category"):
            reg.build("existing", category="missing_category")

    def test_registry_with_category(self) -> None:
        """Register same key in different categories."""
        reg = Registry("test_cat")

        @reg.register("block", category="resnet")
        class ResNetBlock(DummyModel):
            pass

        @reg.register("block", category="vit")
        class ViTBlock(DummyModel):
            pass

        res = reg.build("block", category="resnet")
        assert isinstance(res, DummyModel)

        vit = reg.build("block", category="vit")
        assert isinstance(vit, DummyModel)

        # Different instances
        assert res is not vit

    def test_keys_filtered_by_category(self) -> None:
        """keys() with category filter returns only that category's keys."""
        reg = Registry("test_keys")

        @reg.register("a", category="cat1")
        class A:
            pass

        @reg.register("b", category="cat1")
        class B:
            pass

        @reg.register("c", category="cat2")
        class C:
            pass

        cat1_keys = reg.keys("cat1")
        assert cat1_keys == ["a", "b"]

        cat2_keys = reg.keys("cat2")
        assert cat2_keys == ["c"]

        # Unknown category returns empty list
        assert reg.keys("nonexistent") == []

    def test_global_registries_are_singletons(self) -> None:
        """get_registry returns same instance as global variables."""
        # Verify each global registry can be retrieved
        assert Registry.get_registry("activation") is ACTIVATION_REGISTRY
        assert Registry.get_registry("normalization") is NORMALIZATION_REGISTRY
        assert Registry.get_registry("pooling") is POOLING_REGISTRY
        assert Registry.get_registry("block") is BLOCK_REGISTRY
        assert Registry.get_registry("model") is MODEL_REGISTRY
        assert Registry.get_registry("loss") is LOSS_REGISTRY

    def test_get_registry_key_error(self) -> None:
        """get_registry with non-existent name raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            Registry.get_registry("non_existent_registry")

    def test_contains(self) -> None:
        """contains() and __contains__ work correctly."""
        reg = Registry("test_contains")

        @reg.register("key_a")
        class A:
            pass

        assert reg.contains("key_a") is True
        assert reg.contains("key_b") is False
        assert reg.contains("key_a", category="missing_cat") is False
        assert "key_a" in reg
        assert "key_b" not in reg

    def test_keys_no_category(self) -> None:
        """keys() without category returns default-category keys."""
        reg = Registry("test_keys_no_cat")

        @reg.register("x")
        class X:
            pass

        @reg.register("y")
        class Y:
            pass

        assert reg.keys() == ["x", "y"]

    def test_name_property(self) -> None:
        """name property returns the registry name."""
        reg = Registry("my_reg")
        assert reg.name == "my_reg"

    def test_repr(self) -> None:
        """__repr__ is informative."""
        reg = Registry("repr_test")
        rep = repr(reg)
        assert "repr_test" in rep
