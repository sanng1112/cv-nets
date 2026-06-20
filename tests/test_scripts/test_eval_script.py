"""
Tests for the evaluation script ``scripts/evaluate.py``.

Covers argument parsing and the main function (mocked).
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")

from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest
import torch

from scripts.evaluate import _parse_args, main


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Tests for ``_parse_args()``."""

    def test_checkpoint_required(self) -> None:
        """Without --checkpoint, the parser should exit."""
        with pytest.raises(SystemExit):
            _parse_args([])

    def test_data_root_required(self) -> None:
        """Without --data-root, the parser should exit."""
        with pytest.raises(SystemExit):
            _parse_args(["--checkpoint", "model.pt"])

    def test_minimal_args(self) -> None:
        """Minimal valid arguments produce a namespace."""
        args = _parse_args([
            "--checkpoint", "model.pt",
            "--data-root", "/data",
        ])
        assert args.checkpoint == "model.pt"
        assert args.data_root == "/data"
        assert args.batch_size == 64  # default
        assert args.device is None  # default
        assert args.workers == 0  # default

    def test_all_args_parsed(self) -> None:
        """All optional arguments are parsed correctly."""
        argv = [
            "--checkpoint", "ckpt.pt",
            "--data-root", "/data/test",
            "--batch-size", "32",
            "--device", "cuda",
            "--workers", "4",
        ]
        args = _parse_args(argv)
        assert args.checkpoint == "ckpt.pt"
        assert args.data_root == "/data/test"
        assert args.batch_size == 32
        assert args.device == "cuda"
        assert args.workers == 4


# ---------------------------------------------------------------------------
# main (mocked)
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the ``main()`` function with mocked internals."""

    @mock.patch("scripts.evaluate.torch.load")
    @mock.patch("scripts.evaluate.MODEL_REGISTRY")
    @mock.patch("scripts.evaluate._build_eval_dataset")
    @mock.patch("scripts.evaluate.build_dataloader")
    @mock.patch("scripts.evaluate._run_validation")
    @mock.patch("scripts.evaluate.nn.CrossEntropyLoss")
    def test_main_success(
        self,
        mock_loss_cls,
        mock_run_val,
        mock_build_dataloader,
        mock_build_dataset,
        mock_model_registry,
        mock_torch_load,
    ) -> None:
        """main() should return 0 on success."""
        # Mock checkpoint data
        mock_state_dict = {"fc.weight": torch.randn(10, 4), "fc.bias": torch.randn(10)}
        mock_torch_load.return_value = {
            "model_state_dict": mock_state_dict,
            "model_name": "resnet18",
        }

        # Mock model registry
        mock_model = mock.MagicMock(spec=torch.nn.Module)
        mock_model_registry.contains.return_value = True
        mock_model_registry.build.return_value = mock_model

        # Mock other components
        mock_build_dataset.return_value = mock.MagicMock(spec=object)
        mock_build_dataloader.return_value = mock.MagicMock(spec=object)
        mock_run_val.return_value = {"accuracy": 0.95, "avg_loss": 0.1}
        mock_loss_cls.return_value = mock.MagicMock(spec=object)

        exit_code = main([
            "--checkpoint", "model.pt",
            "--data-root", "/data",
        ])
        assert exit_code == 0

    @mock.patch("scripts.evaluate.torch.load")
    def test_main_failure(self, mock_torch_load) -> None:
        """main() should return 1 when an exception occurs."""
        mock_torch_load.side_effect = RuntimeError("Load failed")

        exit_code = main([
            "--checkpoint", "model.pt",
            "--data-root", "/data",
        ])
        assert exit_code == 1
