"""Tests for BaseDataset abstract class."""

import pytest
import torch
from torch import Tensor
from cvnets.data import BaseDataset


class TestBaseDataset:
    def test_cannot_instantiate_base_dataset(self):
        """BaseDataset is abstract — must raise TypeError."""
        with pytest.raises(TypeError):
            BaseDataset()  # noqa: E1120

    def test_concrete_subclass_must_implement_len_and_getitem(self):
        """Subclass must implement __len__ and __getitem__."""
        class GoodDataset(BaseDataset):
            def __len__(self):
                return 10
            def __getitem__(self, idx):
                return torch.tensor(idx)

        ds = GoodDataset()
        assert len(ds) == 10
        item = ds[3]
        assert isinstance(item, Tensor)
        assert item.item() == 3

    def test_missing_len_raises_type_error(self):
        """Subclass without __len__ must raise TypeError."""
        with pytest.raises(TypeError):
            class BadLen(BaseDataset):
                def __getitem__(self, idx):
                    return idx
            BadLen()  # noqa: E1120

    def test_missing_getitem_raises_type_error(self):
        """Subclass without __getitem__ must raise TypeError."""
        with pytest.raises(TypeError):
            class BadGetItem(BaseDataset):
                def __len__(self):
                    return 5
            BadGetItem()  # noqa: E1120

    def test_registry_available(self):
        """DATA_REGISTRY singleton is accessible."""
        from cvnets.data import DATA_REGISTRY, register_dataset
        assert DATA_REGISTRY.name == "dataset"
        # register_dataset returns a decorator
        decorator = register_dataset("test_ds")
        assert callable(decorator)


class TestRegisterDataset:
    def test_decorator_registers_class(self):
        """@register_dataset adds class to DATA_REGISTRY."""
        from cvnets.data import DATA_REGISTRY, register_dataset, BaseDataset

        @register_dataset("mock_registry_test")
        class MockDS(BaseDataset):
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return torch.tensor(0)

        assert DATA_REGISTRY.contains("mock_registry_test")
        instance = DATA_REGISTRY.build("mock_registry_test")
        assert isinstance(instance, MockDS)
        assert len(instance) == 1
