"""
Registry — a decorator-based factory registry for reusable components.

Provides a lightweight mechanism to associate string keys (optionally
scoped by *category*) with Python callables and later build them via a
uniform factory interface.  Common registries are exposed as module-level
singletons.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, TypeVar

T = TypeVar("T", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Internal storage
# ---------------------------------------------------------------------------

_registries: Dict[str, "Registry"] = {}


# ===================================================================
# Registry
# ===================================================================

class Registry:
    """A named registry of callable entries, grouped by optional category.

    Typical usage::

        BLOCKS = Registry("blocks")

        @BLOCKS.register("resnet_bottleneck")
        class ResNetBottleneck(BaseBlock):
            ...

        block = BLOCKS.build("resnet_bottleneck", in_channels=64, ...)

    Parameters
    ----------
    name : str
        Unique name for this registry (used for singleton access).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        # _entries: {category: {key: callable}}
        self._entries: Dict[str, Dict[str, Callable[..., Any]]] = {}
        _registries[name] = self

    # -- Properties ---------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the registry name."""
        return self._name

    # -- Registration -------------------------------------------------------

    def register(
        self,
        key: str,
        category: str = "",
    ) -> Callable[[T], T]:
        """Return a decorator that registers the decorated callable.

        Parameters
        ----------
        key : str
            Identifier under which the callable is stored.
        category : str
            Optional grouping key (default ``""`` means no group).

        Returns
        -------
        Callable
            A decorator that registers the callable and returns it unchanged.

        Raises
        ------
        ValueError
            If *key* is already registered in the same *category*.
        """
        if category not in self._entries:
            self._entries[category] = {}

        def decorator(fn: T) -> T:
            if key in self._entries[category]:
                raise ValueError(
                    f"Duplicate registration in registry {self._name!r}: "
                    f"key {key!r} already exists in category {category!r}. "
                    f"Available keys in this category: "
                    f"{list(self._entries[category].keys())}"
                )
            self._entries[category][key] = fn
            return fn

        return decorator

    # -- Factory / build ----------------------------------------------------

    def build(
        self,
        key: str,
        category: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Construct an instance of the registered callable for *key*.

        Parameters
        ----------
        key : str
            Registered identifier.
        category : str
            Category to look in (default ``""``).
        *args, **kwargs
            Forwarded to the registered callable.

        Returns
        -------
        Any
            The return value of the registered callable (typically an
            object instance).

        Raises
        ------
        KeyError
            If the *key* (or *category*) is not found.
        """
        if category not in self._entries:
            raise KeyError(
                f"Category {category!r} not found in registry {self._name!r}. "
                f"Available categories: {list(self._entries.keys())}"
            )
        if key not in self._entries[category]:
            raise KeyError(
                f"Key {key!r} not found in registry {self._name!r} "
                f"(category {category!r}). "
                f"Available keys: {list(self._entries[category].keys())}"
            )
        return self._entries[category][key](*args, **kwargs)

    # -- Query --------------------------------------------------------------

    def keys(self, category: str = "") -> List[str]:
        """Return a sorted list of registered keys in *category*.

        If *category* does not exist, returns an empty list.
        """
        if category not in self._entries:
            return []
        return sorted(self._entries[category].keys())

    def contains(self, key: str, category: str = "") -> bool:
        """Return ``True`` if *key* exists in *category*."""
        return (
            category in self._entries and key in self._entries[category]
        )

    def __contains__(self, key: str) -> bool:
        """Convenience — check *key* in the empty-string category."""
        return self.contains(key, category="")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self._name!r}, "
            f"categories={list(self._entries.keys())})"
        )

    # -- Class-level singleton access ---------------------------------------

    @classmethod
    def get_registry(cls, name: str) -> Registry:
        """Return the ``Registry`` singleton previously created with *name*.

        Raises
        ------
        KeyError
            If no registry with that *name* exists.
        """
        if name not in _registries:
            raise KeyError(
                f"Registry {name!r} not found. "
                f"Available registries: {list(_registries.keys())}"
            )
        return _registries[name]


# ===================================================================
# Global singleton registries
# ===================================================================

ACTIVATION_REGISTRY = Registry("activation")
"""Registry for activation function layers."""

NORMALIZATION_REGISTRY = Registry("normalization")
"""Registry for normalization layers (e.g. BatchNorm, LayerNorm)."""

POOLING_REGISTRY = Registry("pooling")
"""Registry for pooling layers (e.g. MaxPool, AvgPool, GlobalPool)."""

BLOCK_REGISTRY = Registry("block")
"""Registry for higher-level building blocks (e.g. ResNet bottleneck)."""

MODEL_REGISTRY = Registry("model")
"""Registry for full model architectures."""

LOSS_REGISTRY = Registry("loss")
"""Registry for loss functions."""
