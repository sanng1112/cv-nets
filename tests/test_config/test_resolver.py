"""
Tests for the ``ConfigResolver`` class.
"""

from __future__ import annotations

import pathlib
import types
from typing import Any, Dict

import pytest
import yaml

from cvnets.config.resolver import ConfigResolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dict() -> Dict[str, Any]:
    return {
        "model": {
            "name": "test_model",
            "layers": [
                {"type": "conv2d", "in_channels": 3, "out_channels": 64},
                {"type": "relu"},
                {"type": "conv2d", "in_channels": 64, "out_channels": 128},
            ],
            "params": {"dropout": 0.5, "use_bias": True},
        },
        "train": {
            "epochs": 100,
            "batch_size": 32,
            "optimizer": {"name": "adam", "lr": 0.001},
        },
    }


# ===================================================================
# TestConfigResolver
# ===================================================================

class TestConfigResolver:
    """Test suite for ConfigResolver."""

    def test_from_dict(self, sample_dict: Dict[str, Any]) -> None:
        """Create from dict, verify get() and dotted paths."""
        resolver = ConfigResolver(sample_dict)
        assert resolver.get("model") == sample_dict["model"]
        assert resolver.get("train") == sample_dict["train"]
        assert resolver.get("model.name") == "test_model"
        assert resolver.get("model.layers.0.type") == "conv2d"
        assert resolver.get("model.layers.1.type") == "relu"
        assert resolver.get("model.layers.2.in_channels") == 64
        assert resolver.get("train.optimizer.name") == "adam"
        assert resolver.get("train.optimizer.lr") == 0.001
        assert resolver["model.name"] == "test_model"
        assert resolver["model.layers.0.type"] == "conv2d"

    def test_missing_key_returns_default(self, sample_dict: Dict[str, Any]) -> None:
        """get(\"nonexistent\") returns default without raising."""
        resolver = ConfigResolver(sample_dict)
        assert resolver.get("nonexistent") is None
        assert resolver.get("nonexistent", 42) == 42
        assert resolver.get("model.nonexistent") is None
        assert resolver.get("model.nonexistent", "fallback") == "fallback"

    def test_missing_key_raises_keyerror(self, sample_dict: Dict[str, Any]) -> None:
        """__getitem__ with missing key raises KeyError."""
        resolver = ConfigResolver(sample_dict)
        with pytest.raises(KeyError, match="not found"):
            _ = resolver["model.nonexistent"]
        with pytest.raises(KeyError, match="not found"):
            _ = resolver["nonexistent_top"]

    def test_to_namespace(self, sample_dict: Dict[str, Any]) -> None:
        """Verify recursive namespace conversion for nested dicts and lists."""
        resolver = ConfigResolver(sample_dict)
        ns = resolver.to_namespace()
        assert isinstance(ns, types.SimpleNamespace)
        assert isinstance(ns.model, types.SimpleNamespace)
        assert isinstance(ns.train, types.SimpleNamespace)
        assert ns.model.name == "test_model"
        assert ns.model.params.dropout == 0.5
        assert ns.train.epochs == 100
        assert ns.train.optimizer.name == "adam"
        assert isinstance(ns.model.layers, list)
        assert len(ns.model.layers) == 3
        assert isinstance(ns.model.layers[0], types.SimpleNamespace)
        assert ns.model.layers[0].type == "conv2d"
        assert ns.model.layers[0].in_channels == 3
        assert ns.model.layers[2].out_channels == 128

    def test_merge_overrides(self, sample_dict: Dict[str, Any]) -> None:
        """Verify deep merge preserves unchanged keys."""
        resolver = ConfigResolver(sample_dict)
        overrides = {
            "model": {
                "params": {"dropout": 0.3},
                "layers": [{"type": "conv2d", "in_channels": 3, "out_channels": 32}],
            },
            "train": {"optimizer": {"lr": 0.0001}},
        }
        merged = resolver.merge(overrides)
        assert resolver.get("model.params.dropout") == 0.5
        assert merged.get("model.params.dropout") == 0.3
        assert merged.get("train.optimizer.lr") == 0.0001
        assert merged.get("model.name") == "test_model"
        assert merged.get("train.epochs") == 100
        assert merged.get("train.batch_size") == 32
        assert len(merged.get("model.layers")) == 1
        assert merged.get("model.layers.0.out_channels") == 32

    def test_from_yaml(self, tmp_path: pathlib.Path, sample_dict: Dict[str, Any]) -> None:
        """Create a temp yaml, load it, and verify content."""
        yaml_path = tmp_path / "test_config.yaml"
        with yaml_path.open("w", encoding="utf-8") as fh:
            yaml.dump(sample_dict, fh, default_flow_style=False)
        resolver = ConfigResolver(yaml_path)
        assert resolver.get("model.name") == "test_model"
        assert resolver.get("model.layers.0.type") == "conv2d"
        assert resolver.get("train.epochs") == 100
        assert resolver.get("train.optimizer.lr") == 0.001
        resolver_str = ConfigResolver(str(yaml_path))
        assert resolver_str.get("model.name") == "test_model"

    def test_contains(self, sample_dict: Dict[str, Any]) -> None:
        """in operator works for dotted paths."""
        resolver = ConfigResolver(sample_dict)
        assert "model" in resolver
        assert "model.name" in resolver
        assert "model.layers.0.type" in resolver
        assert "nonexistent" not in resolver
        assert "model.nonexistent" not in resolver

    def test_to_dict_deep_copy(self, sample_dict: Dict[str, Any]) -> None:
        """to_dict returns a deep copy; mutations don't affect original."""
        resolver = ConfigResolver(sample_dict)
        exported = resolver.to_dict()
        exported["model"]["name"] = "mutated"
        exported["model"]["params"]["dropout"] = 0.0
        assert resolver.get("model.name") == "test_model"
        assert resolver.get("model.params.dropout") == 0.5

    def test_to_yaml(self, tmp_path: pathlib.Path, sample_dict: Dict[str, Any]) -> None:
        """to_yaml writes a valid YAML file that can be read back."""
        resolver = ConfigResolver(sample_dict)
        out_path = tmp_path / "output.yaml"
        resolver.to_yaml(out_path)
        with out_path.open("r", encoding="utf-8") as fh:
            reloaded = yaml.safe_load(fh)
        assert reloaded == sample_dict

    def test_from_simple_namespace(self, sample_dict: Dict[str, Any]) -> None:
        """Construct from SimpleNamespace works correctly."""
        ns = ConfigResolver.dict_to_namespace(sample_dict)
        resolver = ConfigResolver(ns)
        assert resolver.get("model.name") == "test_model"
        assert resolver.get("train.optimizer.lr") == 0.001

    def test_from_none(self) -> None:
        """Construct with None gives empty config."""
        resolver = ConfigResolver(None)
        assert resolver.to_dict() == {}

    @staticmethod
    def test_static_helpers() -> None:
        """Static helpers namespace_to_dict and dict_to_namespace roundtrip."""
        data = {"a": 1, "b": {"c": [1, 2, {"d": 3}]}}
        ns = ConfigResolver.dict_to_namespace(data)
        assert isinstance(ns, types.SimpleNamespace)
        assert ns.a == 1
        assert ns.b.c[0] == 1
        assert ns.b.c[2].d == 3
        back = ConfigResolver.namespace_to_dict(ns)
        assert back == data

    def test_merge_returns_new_resolver(self, sample_dict: Dict[str, Any]) -> None:
        """merge() returns a new resolver; original not modified."""
        resolver = ConfigResolver(sample_dict)
        merged = resolver.merge({"model": {"name": "overridden"}})
        assert resolver.get("model.name") == "test_model"
        assert merged.get("model.name") == "overridden"
        assert resolver is not merged

    def test_yaml_no_file_raises(self) -> None:
        """Loading from a non-existent YAML path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ConfigResolver("/nonexistent/path/config.yaml")

    def test_invalid_source_type(self) -> None:
        """Passing an invalid source type raises TypeError."""
        with pytest.raises(TypeError):
            ConfigResolver(123)  # type: ignore[arg-type]