"""
Activation layer system — standalone implementation with dual registration.

Mirrors the API of the legacy ``layers.activation`` package while also
registering every activation function into the global
:obj:`ACTIVATION_REGISTRY <cvnets.core.registry.ACTIVATION_REGISTRY>`.

Exports
-------
SUPPORTED_ACT_FNS : list[str]
ACT_FN_MODULES : dict[str, type]
register_act_fn(name)
build_activation_layer(opts=None, act_type=None, **kwargs)
get_config_prop(opts, prop_path, default=None)
"""

from __future__ import annotations

import importlib
import inspect
import os
from typing import Any, Dict, List, Optional, Union
from types import SimpleNamespace

import torch.nn as nn

from cvnets.core.registry import ACTIVATION_REGISTRY

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

SUPPORTED_ACT_FNS: List[str] = []
ACT_FN_MODULES: Dict[str, type] = {}


def register_act_fn(name: str):
    """Decorator: register *cls* under *name* in both legacy dicts and
    ``ACTIVATION_REGISTRY``.
    """
    def decorator(cls):
        if name in SUPPORTED_ACT_FNS:
            raise ValueError(
                f"Cannot register duplicate activation function ({name})"
            )
        SUPPORTED_ACT_FNS.append(name)
        ACT_FN_MODULES[name] = cls
        ACTIVATION_REGISTRY.register(name)(cls)
        return cls
    return decorator


def get_config_prop(
    opts: Any,
    prop_path: str,
    default: Any = None,
) -> Any:
    """Access a nested attribute / key from *opts* following a dotted path."""
    try:
        parts = prop_path.split(".")
        current = opts
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part)
        return current if current is not None else default
    except (AttributeError, KeyError, TypeError):
        return default


def build_activation_layer(
    opts: Optional[Union[dict, Any]] = None,
    act_type: Optional[str] = None,
    inplace: Optional[bool] = None,
    negative_slope: Optional[float] = None,
    num_parameters: Optional[int] = None,
    **kwargs: Any,
) -> Optional[nn.Module]:
    """Construct and return an activation layer.

    Resolves the activation type (explicit argument > *opts*), collects
    common parameters, filters them through the class constructor signature,
    and instantiates the module.
    """
    # 1. Resolve type
    if act_type is None:
        if isinstance(opts, dict):
            act_type = opts.get("type")
        else:
            act_type = (
                getattr(opts, "type", None) if opts is not None else None
            )
    if not act_type:
        return None

    # 2. Resolve common params
    if inplace is None:
        inplace = (
            opts.get("inplace", False)
            if isinstance(opts, dict)
            else getattr(opts, "inplace", False)
        )
    if negative_slope is None:
        if isinstance(opts, dict):
            negative_slope = opts.get("neg_slope") or opts.get(
                "neg-slope", 0.1
            )
        else:
            negative_slope = getattr(
                opts, "neg_slope", getattr(opts, "neg-slope", 0.1)
            )
    if num_parameters is None:
        num_parameters = (
            opts.get("num_parameters", 1)
            if isinstance(opts, dict)
            else getattr(opts, "num_parameters", 1)
        )

    act_type = act_type.lower()

    if act_type not in SUPPORTED_ACT_FNS:
        from cvnets.utils.logger import error as _log_error

        _log_error(
            f"Supported activation layers: {SUPPORTED_ACT_FNS}. "
            f"Supplied: {act_type}"
        )
        raise NotImplementedError(
            f"Activation function '{act_type}' is not supported/registered."
        )

    act_class = ACT_FN_MODULES[act_type]

    # 3. Collect candidate params
    raw_args: Dict[str, Any] = {
        "inplace": inplace,
        "negative_slope": negative_slope,
        "num_parameters": num_parameters,
    }
    raw_args.update(kwargs)

    # 4. Filter to match constructor signature
    sig = inspect.signature(act_class.__init__)
    allowed = sig.parameters

    filtered: Dict[str, Any] = {}
    for pname, param in allowed.items():
        if pname in ("self", "args", "kwargs"):
            continue
        if pname in raw_args:
            filtered[pname] = raw_args[pname]

    if any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in allowed.values()
    ):
        filtered.update(raw_args)

    return act_class(**filtered)


# ---------------------------------------------------------------------------
# Auto-import all sibling modules
# ---------------------------------------------------------------------------

_act_dir = os.path.dirname(__file__)
for _f in sorted(os.listdir(_act_dir)):
    _path = os.path.join(_act_dir, _f)
    if _f.startswith("_") or _f.startswith("."):
        continue
    if _f.endswith(".py") or os.path.isdir(_path):
        _module_name = _f[: _f.find(".py")] if _f.endswith(".py") else _f
        try:
            importlib.import_module(
                ".{}".format(_module_name), package=__package__
            )
        except Exception:
            pass
