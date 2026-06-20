"""Integration test: CIFAR10 + transforms + DataLoader end-to-end.

Uses synthetic CIFAR-10 data to avoid network dependency.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Subset

from cvnets.data import build_dataloader
from cvnets.data.datasets import CIFAR10
from cvnets.utils import (
    double_dash_line,
    info,
    print_header,
    singe_dash_line,
)


def _create_synthetic_cifar10(root: str, num_train: int = 50, num_test: int = 10):
    """Create a minimal CIFAR-10-like dataset for testing."""
    batch_dir = Path(root) / "cifar-10-batches-py"
    batch_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)
    for i in range(1, 6):
        data = rng.randint(0, 256, size=(num_train, 3072), dtype=np.uint8)
        labels = rng.randint(0, 10, size=(num_train,)).tolist()
        with open(batch_dir / f"data_batch_{i}", "wb") as f:
            pickle.dump({"data": data, "labels": labels}, f)
    data = rng.randint(0, 256, size=(num_test, 3072), dtype=np.uint8)
    labels = rng.randint(0, 10, size=(num_test,)).tolist()
    with open(batch_dir / "test_batch", "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)
    return str(batch_dir)


class TestCIFAR10Pipeline:
    """End-to-end pipeline: dataset -> transforms -> dataloader."""

    TRAIN_TRANSFORM = [
        {"type": "random_horizontal_flip", "p": 0.5},
    ]
    VAL_TRANSFORM = []

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.root = str(tmp_path / "cifar10_integration")
        _create_synthetic_cifar10(self.root, num_train=50, num_test=10)

    def setup_method(self):
        print_header("CIFAR10 Pipeline Integration Test")

    def teardown_method(self):
        double_dash_line()

    def test_train_dataloader_yields_correct_shapes(self):
        info("Loading CIFAR10 training set with random flip")
        ds = CIFAR10(
            root=self.root,
            train=True,
            transform=self.TRAIN_TRANSFORM,
        )
        dl = build_dataloader(ds, batch_size=16, shuffle=True, num_workers=0, verbose=False)
        batch = next(iter(dl))
        images, labels = batch

        assert images.shape == (16, 3, 32, 32)
        assert images.dtype == torch.float32
        assert labels.shape == (16,)
        info(f"Train batch OK - images: {images.shape}, labels: {labels.shape}")
        singe_dash_line()

    def test_val_dataloader_yields_correct_shapes(self):
        info("Loading CIFAR10 validation set")
        ds = CIFAR10(
            root=self.root,
            train=False,
        )
        dl = build_dataloader(ds, batch_size=8, shuffle=False, verbose=False)
        batch = next(iter(dl))
        images, labels = batch

        assert images.shape == (8, 3, 32, 32)
        assert labels.shape == (8,)
        info(f"Val batch OK - images: {images.shape}, labels: {labels.shape}")
        singe_dash_line()

    def test_pipeline_is_iterable_full_epoch(self):
        ds = CIFAR10(
            root=self.root,
            train=True,
        )
        # Use small subset for speed
        subset = Subset(ds, range(32))
        dl = build_dataloader(subset, batch_size=8, verbose=False)

        count = 0
        for images, labels in dl:
            count += images.size(0)
            assert labels.size(0) == images.size(0)

        assert count == 32
        info(f"Full epoch iteration OK - {count} samples processed")

    def test_output_matches_trainer_signature(self):
        """Confirm data output matches Trainer's expected input."""
        ds = CIFAR10(
            root=self.root,
            train=True,
        )
        dl = build_dataloader(ds, batch_size=4, verbose=False)

        for batch_idx, (inputs, targets) in enumerate(dl):
            if batch_idx >= 2:
                break
            assert inputs.dim() == 4  # (B, C, H, W)
            assert targets.dim() == 1  # (B,)

        info("Trainer signature compatibility OK")
