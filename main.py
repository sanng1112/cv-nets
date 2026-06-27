"""
CV-Nets: Training Dispatcher

This is the legacy entry point. It now delegates to ``scripts.train.main``
which uses the canonical ``cvnets`` package (``src/cvnets/``).

For new work, use the CLI entry points directly::

    cvnets-train --config configs/demo.yaml
    cvnets-eval --checkpoint <path>

Or run the scripts directly::

    uv run python scripts/train.py --config configs/demo.yaml
"""

from __future__ import annotations

import sys
import warnings

warnings.warn(
    "main.py is deprecated. Use `cvnets-train` or `uv run python scripts/train.py` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from scripts.train import main as train_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(train_main())
