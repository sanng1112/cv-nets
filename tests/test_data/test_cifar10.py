"""Tests for CIFAR10 dataset.

Uses synthetic CIFAR-10 format data to avoid network dependency.
"""

import os
import pickle
import struct
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from cvnets.data.registry import DATA_REGISTRY
from cvnets.data.datasets import CIFAR10


def _create_synthetic_cifar10(root: str, num_train: int = 5, num_test: int = 2):
    """Create a minimal CIFAR-10-like dataset for testing."""
    batch_dir = Path(root) / "cifar-10-batches-py"
    batch_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)

    for i in range(1, 6):
        data = rng.randint(0, 256, size=(num_train, 3072), dtype=np.uint8)
        labels = rng.randint(0, 10, size=(num_train,)).tolist()
        with open(batch_dir / f"data_batch_{i}", "wb") as f:
            pickle.dump({"data": data, "labels": labels}, f)

    # Test batch
    data = rng.randint(0, 256, size=(num_test, 3072), dtype=np.uint8)
    labels = rng.randint(0, 10, size=(num_test,)).tolist()
    with open(batch_dir / "test_batch", "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)

    return str(batch_dir)


class TestCIFAR10Dataset:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.data_dir = _create_synthetic_cifar10(str(tmp_path / "cifar10"))
        self.root = str(Path(self.data_dir).parent)

    def test_registered_in_data_registry(self):
        assert DATA_REGISTRY.contains("cifar10")

    def test_len_with_synthetic_data(self):
        ds = CIFAR10(root=self.root, train=True)
        assert len(ds) == 5 * 5  # 5 batches x 5 samples

    def test_len_test_split(self):
        ds = CIFAR10(root=self.root, train=False)
        assert len(ds) == 2  # 2 test samples

    def test_getitem_returns_tuple_of_tensor_and_int(self):
        ds = CIFAR10(root=self.root, train=True)
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, int)
        assert img.shape == (3, 32, 32)

    def test_classes_property(self):
        ds = CIFAR10(root=self.root, train=True)
        assert len(ds.classes) == 10
        assert ds.classes[0] == "airplane"

    def test_download_missing_data_raises(self, tmp_path):
        """Without download=True, missing data raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            CIFAR10(root=str(tmp_path / "nonexistent"), train=True, download=False)
