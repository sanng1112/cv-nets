"""Auto-import all classification loss functions."""
from __future__ import annotations
import importlib
import os
for _f in sorted(os.listdir(os.path.dirname(__file__))):
    if _f.endswith(".py") and not _f.startswith("_"):
        try:
            importlib.import_module(f".{_f[:-3]}", package=__package__)
        except Exception:
            pass
