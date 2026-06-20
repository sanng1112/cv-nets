"""
Tests for the training script ``scripts/train.py``.

Covers argument parsing, config merging, and the main function (mocked).
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")

import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest
import torch
import yaml

from scripts.train import _parse_args, _merge_args_with_config, main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Return a minimal training configuration YAML dict."""
    return {
        "model": {
            "name": "resnet18",
            "zoo_kwargs": {"num_classes": 10},
        },
        "data": {
            "dataset": "cifar10",
            "root": "./data",
            "batch_size": 32,
            "workers": 2,
        },
        "train": {
            "epochs": 5,
            "lr": 0.01,
            "optimizer": "sgd",
            "loss": "cross_entropy",
            "seed": 123,
        },
    }

# ---------------------------------------------------------------------------
# _merge_args_with_config
# ---------------------------------------------------------------------------


class TestMergeArgsWithConfig:
    """Tests for ``_merge_args_with_config()``."""

    def test_cli_overrides_config(self, sample_config: Dict[str, Any]) -> None:
        """CLI arguments take precedence over config file values."""
        args = _parse_args([
            "--config", "dummy",
            "--epochs", "100",
            "--lr", "0.0001",
            "--batch-size", "256",
            "--device", "cpu",
            "--amp",
        ])
        merged = _merge_args_with_config(args, sample_config)

        assert merged["train"]["epochs"] == 100
        assert merged["train"]["lr"] == 0.0001
        assert merged["data"]["batch_size"] == 256
        assert merged["train"]["device"] == "cpu"
        assert merged["train"]["amp"] is True

    def test_config_defaults_used_when_no_cli(
        self, sample_config: Dict[str, Any]
    ) -> None:
        """Config values are preserved when CLI args are not provided."""
        args = _parse_args(["--config", "dummy"])
        merged = _merge_args_with_config(args, sample_config)

        assert merged["train"]["epochs"] == 5
        assert merged["train"]["lr"] == 0.01
        assert merged["data"]["batch_size"] == 32
        assert merged["train"]["seed"] == 123

    def test_defaults_filled_when_missing(
        self, sample_config: Dict[str, Any]
    ) -> None:
        """Defaults are applied for keys missing from both CLI and config."""
        del sample_config["train"]["epochs"]
        del sample_config["train"]["lr"]
        del sample_config["data"]["batch_size"]

        args = _parse_args(["--config", "dummy"])
        merged = _merge_args_with_config(args, sample_config)

        assert merged["train"]["epochs"] == 10  # default
        assert merged["train"]["lr"] == 0.001  # default
        assert merged["data"]["batch_size"] == 64  # default


# ---------------------------------------------------------------------------
# main (mocked)
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the ``main()`` function with mocked internals."""

    @mock.patch("scripts.train.safe_load_yaml")
    @mock.patch("scripts.train._build_model")
    @mock.patch("scripts.train._build_dataset")
    @mock.patch("scripts.train.build_dataloader")
    @mock.patch("scripts.train._build_criterion")
    @mock.patch("scripts.train.build_optimizer")
    @mock.patch("scripts.train.Trainer")
    def test_main_success(
        self,
        mock_trainer_cls,
        mock_build_optimizer,
        mock_build_criterion,
        mock_build_dataloader,
        mock_build_dataset,
        mock_build_model,
        mock_safe_load_yaml,
        sample_config,
    ) -> None:
        """main() should return 0 on success."""
        mock_safe_load_yaml.return_value = sample_config
        mock_build_model.return_value = mock.MagicMock(spec=torch.nn.Module)
        mock_build_dataset.return_value = mock.MagicMock(spec=object)
        mock_build_dataloader.return_value = mock.MagicMock(spec=object)
        mock_build_criterion.return_value = mock.MagicMock(spec=object)
        mock_build_optimizer.return_value = mock.MagicMock(spec=object)

        mock_trainer_instance = mock.MagicMock()
        mock_trainer_instance.fit.return_value = {"accuracy": 0.95, "avg_loss": 0.1}
        mock_trainer_cls.return_value = mock_trainer_instance

        exit_code = main(["--config", "dummy.yaml"])
        assert exit_code == 0

    @mock.patch("scripts.train.safe_load_yaml")
    def test_main_failure(self, mock_safe_load_yaml) -> None:
        """main() should return 1 when an exception occurs."""
        mock_safe_load_yaml.side_effect = RuntimeError("Config load failed")

        exit_code = main(["--config", "dummy.yaml"])
        assert exit_code == 1

