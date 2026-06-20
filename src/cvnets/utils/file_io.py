"""Safe file I/O helpers with proper logging.

Note: imports logger functions directly (not via cvnets.utils) to avoid
circular imports when cvnets.utils.__init__ imports this module.
"""

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from cvnets.utils.logger import error, info


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file with error handling and logging.

    Raises
    ------
    LoggerError
        If the file does not exist (via ``cvnets.utils.error()``).
    yaml.YAMLError
        If the file contains invalid YAML.
    """
    p = Path(path)
    if not p.is_file():
        error(f"YAML file not found: {path}")
    info(f"Loading YAML config from {path}")
    with open(p, "r") as f:
        data = yaml.safe_load(f)
    info(f"YAML loaded: {len(data)} top-level keys")
    return data


def safe_load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file with error handling and logging."""
    p = Path(path)
    if not p.is_file():
        error(f"JSON file not found: {path}")
    info(f"Loading JSON from {path}")
    with open(p, "r") as f:
        data = json.load(f)
    return data


def save_json(data: Dict[str, Any], path: str, indent: int = 2) -> None:
    """Save a dictionary as JSON, creating parent directories."""
    p = Path(path)
    ensure_dir(str(p.parent))
    info(f"Saving JSON to {path}")
    with open(p, "w") as f:
        json.dump(data, f, indent=indent)
