"""
ConfigResolver — dotted-path access to configuration dictionaries.

Supports construction from a YAML file path, a plain ``dict``, a
``SimpleNamespace``, or ``None`` (empty config).  Provides safe,
read-only access through dotted-key notation as well as export
to dictionary, namespace, or YAML file.
"""

from __future__ import annotations

import copy
import io
import pathlib
import types
from typing import Any, Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _namespace_to_dict(namespace: types.SimpleNamespace) -> Dict[str, Any]:
    """Recursively convert a ``SimpleNamespace`` (and lists thereof) to a
    plain ``dict``."""
    if isinstance(namespace, types.SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in namespace.__dict__.items()}
    if isinstance(namespace, list):
        return [_namespace_to_dict(item) for item in namespace]
    return namespace


def _dict_to_namespace(d: Dict[str, Any]) -> types.SimpleNamespace:
    """Recursively convert a plain ``dict`` (and lists thereof) to a
    ``SimpleNamespace``."""
    if isinstance(d, dict):
        return types.SimpleNamespace(
            **{k: _dict_to_namespace(v) for k, v in d.items()}
        )
    if isinstance(d, list):
        return [_dict_to_namespace(item) for item in d]
    return d


# ---------------------------------------------------------------------------
# Private helpers for dotted-path resolution
# ---------------------------------------------------------------------------

def _resolve_dotted(data: Any, dotted: str) -> Any:
    """Walk *data* following the dotted *key* and return the value.

    Raises ``KeyError`` when any segment cannot be found.
    """
    if not dotted:
        raise KeyError("Empty key is not allowed")

    parts = dotted.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(
                    f"Key {part!r} not found in config. Available keys: "
                    f"{list(current.keys())}"
                )
            current = current[part]
        elif isinstance(current, (list, tuple)):
            try:
                index = int(part)
            except ValueError:
                raise KeyError(
                    f"Cannot index list/tuple with non-integer key {part!r}"
                )
            if index < 0 or index >= len(current):
                raise KeyError(
                    f"Index {index} out of bounds for list of length {len(current)}"
                )
            current = current[index]
        else:
            raise KeyError(
                f"Cannot traverse into type {type(current).__name__!r} "
                f"with key {part!r}"
            )
    return current


def _deep_merge(base: Any, overrides: Any) -> Any:
    """Deep-merge *overrides* into *base* and return a new object.

    - If both sides are ``dict``\ s, keys from *overrides* overwrite
      (or are merged into) *base* keys recursively.
    - Otherwise *overrides* wins.
    """
    if not isinstance(base, dict) or not isinstance(overrides, dict):
        return copy.deepcopy(overrides)

    merged = copy.deepcopy(base)
    for key, val in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


# ===================================================================
# ConfigResolver
# ===================================================================

class ConfigResolver:
    """Resolve configuration values via dotted-path keys.

    Parameters
    ----------
    source : str or Path or dict or SimpleNamespace or None
        The initial configuration data.

        * ``str`` / ``Path`` — interpreted as a YAML file path.
        * ``dict`` — used directly (a deep copy is kept).
        * ``SimpleNamespace`` — converted to a dict internally.
        * ``None`` — start with an empty dictionary.
    """

    def __init__(
        self,
        source: Optional[
            Union[str, pathlib.Path, Dict[str, Any], types.SimpleNamespace]
        ] = None,
    ) -> None:
        self._data: Dict[str, Any] = {}

        if source is None:
            self._data = {}
        elif isinstance(source, (str, pathlib.Path)):
            path = pathlib.Path(source)
            with path.open("r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh)
            self._data = loaded if isinstance(loaded, dict) else {}
        elif isinstance(source, types.SimpleNamespace):
            self._data = _namespace_to_dict(source)
        elif isinstance(source, dict):
            self._data = copy.deepcopy(source)
        else:
            raise TypeError(
                f"Unsupported source type: {type(source).__name__!r}. "
                f"Expected str, Path, dict, SimpleNamespace, or None."
            )

    # -- Reading ------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value at *key* (dotted) or *default* if missing."""
        try:
            return _resolve_dotted(self._data, key)
        except (KeyError, IndexError, TypeError, ValueError):
            return default

    def __getitem__(self, key: str) -> Any:
        """Return the value at *key* (dotted) or raise ``KeyError``."""
        return _resolve_dotted(self._data, key)

    def __contains__(self, key: str) -> bool:
        """Return ``True`` if *key* exists (dotted path)."""
        try:
            _resolve_dotted(self._data, key)
            return True
        except (KeyError, IndexError, TypeError, ValueError):
            return False

    # -- Export -------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the internal dictionary."""
        return copy.deepcopy(self._data)

    def to_namespace(self) -> types.SimpleNamespace:
        """Return a recursive ``SimpleNamespace`` representation."""
        return _dict_to_namespace(self._data)

    def to_yaml(self, path: Union[str, pathlib.Path]) -> None:
        """Write the configuration to a YAML file at *path*."""
        path = pathlib.Path(path)
        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(self._data, fh, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={self._data!r})"

    # -- Mutation -----------------------------------------------------------

    def merge(self, overrides: Dict[str, Any]) -> ConfigResolver:
        """Return a new ``ConfigResolver`` with *overrides* deep-merged.

        The original resolver is not modified.
        """
        merged_data = _deep_merge(self._data, overrides)
        return ConfigResolver(merged_data)

    # -- Static helpers -----------------------------------------------------

    @staticmethod
    def namespace_to_dict(
        namespace: types.SimpleNamespace,
    ) -> Dict[str, Any]:
        """Recursively convert *namespace* to a plain dict."""
        return _namespace_to_dict(namespace)

    @staticmethod
    def dict_to_namespace(d: Dict[str, Any]) -> types.SimpleNamespace:
        """Recursively convert a dict to a ``SimpleNamespace``."""
        return _dict_to_namespace(d)
