"""Tests for build_dataloader factory."""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from cvnets.data import build_dataloader


class _DummyDataset(Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, idx):
        return torch.tensor([idx]), torch.tensor(0)


class TestBuildDataloader:
    def test_returns_dataloader(self):
        dl = build_dataloader(_DummyDataset(), batch_size=4, verbose=False)
        assert isinstance(dl, DataLoader)

    def test_batch_size_is_respected(self):
        dl = build_dataloader(_DummyDataset(), batch_size=8, verbose=False)
        batch = next(iter(dl))
        assert batch[0].shape[0] == 8

    def test_shuffle_default(self):
        dl = build_dataloader(_DummyDataset(), batch_size=4, verbose=False)
        # Default sampler is SequentialSampler (not None)
        from torch.utils.data import SequentialSampler
        assert isinstance(dl.sampler, SequentialSampler)

    def test_shuffle_true(self):
        dl = build_dataloader(_DummyDataset(), batch_size=4, shuffle=True, verbose=False)
        from torch.utils.data import RandomSampler
        assert isinstance(dl.sampler, RandomSampler)

    def test_num_workers_passed(self):
        dl = build_dataloader(_DummyDataset(), batch_size=4, num_workers=0, verbose=False)
        assert dl.num_workers == 0

    def test_drop_last_passed(self):
        dl = build_dataloader(_DummyDataset(), batch_size=4, drop_last=True, verbose=False)
        assert dl.drop_last
