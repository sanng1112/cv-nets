"""
ConfigSchema — validation helpers for cv-nets configuration dictionaries.

Provides static methods that validate the structure of model and training
configuration dicts, raising ``ConfigValidationError`` (a subclass of
``ConfigError``) on failure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Avoid triggering eager imports of torch-dependent base modules.
# Access ConfigError directly from the exceptions module.
from cvnets.core.exceptions import ConfigError  # noqa: E402


class ConfigValidationError(ConfigError):
    """Raised when a configuration dictionary fails schema validation."""

    pass


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

_REQUIRED_MODEL_KEYS: List[str] = ["layers"]
_LAYER_REQUIRED_FIELDS: List[str] = ["type"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ConfigSchema:
    """Static validation methods for configuration dictionaries."""

    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> None:
        """Validate a model configuration dictionary.

        Checks:
            1. A ``"model"`` section exists in *config*.
            2. The ``"model"`` section has a ``"layers"`` key whose value
               is a list.
            3. Every layer in the list is a dict containing a ``"type"``
               field.

        Parameters
        ----------
        config : dict
            The full configuration dictionary (top-level keys such as
            ``"model"``, ``"train"``, …).

        Raises
        ------
        ConfigValidationError
            If any of the checks fail.
        """
        if not isinstance(config, dict):
            raise ConfigValidationError(
                f"Configuration must be a dict, got {type(config).__name__!r}."
            )

        # 1. Top-level "model" section
        if "model" not in config:
            raise ConfigValidationError(
                "Configuration is missing the required 'model' section."
            )
        model_cfg = config["model"]
        if not isinstance(model_cfg, dict):
            raise ConfigValidationError(
                f"The 'model' section must be a dict, got {type(model_cfg).__name__!r}."
            )

        # 2. "layers" key in model section
        if "layers" not in model_cfg:
            raise ConfigValidationError(
                "The 'model' section is missing the required 'layers' key."
            )
        layers = model_cfg["layers"]
        if not isinstance(layers, list):
            raise ConfigValidationError(
                f"The 'model.layers' must be a list, got {type(layers).__name__!r}."
            )

        # 3. Each layer entry
        for idx, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise ConfigValidationError(
                    f"Layer at index {idx} is not a dict "
                    f"(got {type(layer).__name__!r})."
                )
            if "type" not in layer:
                raise ConfigValidationError(
                    f"Layer at index {idx} is missing the required 'type' field. "
                    f"Available keys: {list(layer.keys())}"
                )
            if not isinstance(layer["type"], str):
                raise ConfigValidationError(
                    f"Layer 'type' at index {idx} must be a string, "
                    f"got {type(layer['type']).__name__!r}."
                )

    @staticmethod
    def validate_train_config(config: Dict[str, Any]) -> None:
        """Optionally validate a training configuration dictionary.

        Currently this performs a light-weight check that the ``"train"``
        section, if present, contains at least a ``"loss"`` key.

        Parameters
        ----------
        config : dict
            The full configuration dictionary.

        Raises
        ------
        ConfigValidationError
            If the training section is present but structurally invalid.
        """
        if not isinstance(config, dict):
            raise ConfigValidationError(
                f"Configuration must be a dict, got {type(config).__name__!r}."
            )

        if "train" not in config:
            # Training config is optional — nothing to validate.
            return

        train_cfg = config["train"]
        if not isinstance(train_cfg, dict):
            raise ConfigValidationError(
                f"The 'train' section must be a dict, got "
                f"{type(train_cfg).__name__!r}."
            )

        if "loss" not in train_cfg:
            raise ConfigValidationError(
                "The 'train' section is missing the required 'loss' key."
            )
