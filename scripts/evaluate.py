#!/usr/bin/env python3
"""
Evaluation script for cv-nets.

Usage
-----
    cvnets-eval --checkpoint model.pt --data-root /path/to/data
    cvnets-eval --checkpoint model.pt --data-root /path/to/data --batch-size 32
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cvnets.core.registry import MODEL_REGISTRY
from cvnets.data import build_dataloader
from cvnets.data.datasets import CIFAR10, ImageFolderDataset
from cvnets.trainer.metrics import Accuracy, AverageLoss, MetricsTracker
from cvnets.utils import (
    double_dash_line,
    info,
    print_header,
)
from cvnets.utils.logger import error

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        The argument list to parse.  If ``None``, uses ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="cv-nets evaluation script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt or .pth file)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory of the evaluation dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. 'cuda', 'cpu'). Auto-detected if not set.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of DataLoader workers",
    )

    return parser.parse_args(argv)



def _build_eval_dataset(data_root: str) -> torch.utils.data.Dataset:
    """Build an evaluation dataset.

    Tries ``ImageFolderDataset`` first (for generic folder structure),
    then falls back to ``CIFAR10`` test split.
    """
    from pathlib import Path

    root_path = Path(data_root)
    if root_path.is_dir() and any(
        p.is_dir() for p in root_path.iterdir() if not p.name.startswith(".")
    ):
        info(f"Loading ImageFolderDataset from {data_root}")
        return ImageFolderDataset(root=data_root)
    else:
        info(f"Loading CIFAR10 test split from {data_root}")
        return CIFAR10(root=data_root, train=False, download=False)


@torch.no_grad()
def _run_validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run a validation loop and return metrics.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    val_loader : DataLoader
        DataLoader for the validation/test set.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Device to run evaluation on.

    Returns
    -------
    dict
        Dictionary with ``"accuracy"`` and ``"avg_loss"`` keys.
    """
    model.eval()
    tracker = MetricsTracker(
        Accuracy(),
        AverageLoss(),
        metric_names=["accuracy", "avg_loss"],
    )

    for inputs, targets in val_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        tracker.on_batch_end(
            prediction=outputs,
            target=targets,
            loss_value=loss.item(),
            batch_size=inputs.size(0),
        )

    return tracker.compute()


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the evaluation script.

    Parameters
    ----------
    argv : list of str, optional
        The argument list to parse.  If ``None``, uses ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    args = _parse_args(argv)

    try:
        print_header("cv-nets Evaluation Script")
        double_dash_line()

        # ── Device ────────────────────────────────────────────────────
        if args.device is not None:
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        info(f"Using device: {device}")

        # ── Build model from checkpoint ───────────────────────────────
        info(f"Loading checkpoint from {args.checkpoint} ...")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

        # Infer model architecture
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Detect num_classes from classifier weights
        num_classes = 10  # default
        for key in state_dict:
            if any(k in key for k in ("classifier", "fc", "head")) and "weight" in key:
                num_classes = state_dict[key].shape[0]
                break

        info(f"Detected num_classes={num_classes} from checkpoint")

        # Trigger MODEL_REGISTRY
        import cvnets.models.zoo  # noqa: F401

        model_name = checkpoint.get("model_name", "resnet18")

        if MODEL_REGISTRY.contains(model_name):
            info(f"Building model {model_name!r} from MODEL_REGISTRY")
            model = MODEL_REGISTRY.build(model_name, num_classes=num_classes)
        else:
            info(f"Building default resnet18 (num_classes={num_classes})")
            model = MODEL_REGISTRY.build("resnet18", num_classes=num_classes)

        # Load state dict
        model.load_state_dict(state_dict)
        model = model.to(device)
        info("Checkpoint loaded successfully")
        double_dash_line()

        # ── Build dataset & dataloader ────────────────────────────────
        dataset = _build_eval_dataset(args.data_root)
        val_loader: DataLoader = build_dataloader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type != "cpu",
        )
        double_dash_line()

        # ── Criterion ─────────────────────────────────────────────────
        criterion = nn.CrossEntropyLoss()

        # ── Run validation ────────────────────────────────────────────
        info("Running evaluation ...")
        metrics = _run_validation(model, val_loader, criterion, device)

        # ── Print results ─────────────────────────────────────────────
        print_header("Evaluation Results")
        info(f"  Accuracy:  {metrics.get('accuracy', 0.0):.4f}")
        info(f"  Avg Loss:  {metrics.get('avg_loss', 0.0):.4f}")
        double_dash_line()

        return 0

    except Exception as exc:
        error(f"Evaluation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
