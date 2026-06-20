"""Tests for ImageFolderDataset.

Uses PIL to create synthetic test images.
"""

import pytest
from PIL import Image

from cvnets.data.registry import DATA_REGISTRY
from cvnets.data.datasets import ImageFolderDataset


@pytest.fixture
def sample_image_folder(tmp_path):
    """Create a temporary image folder structure with small synthetic images."""
    (tmp_path / "cat").mkdir(parents=True)
    (tmp_path / "dog").mkdir(parents=True)

    for i in range(3):
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        img.save(tmp_path / "cat" / f"cat_{i}.png")

    for i in range(2):
        img = Image.new("RGB", (10, 10), color=(0, 255, 0))
        img.save(tmp_path / "dog" / f"dog_{i}.jpg")

    # Non-image file that should be skipped
    (tmp_path / "cat" / "hidden.txt").write_text("not an image")

    return tmp_path


class TestImageFolderDataset:
    def test_registered(self):
        assert DATA_REGISTRY.contains("image_folder")

    def test_len_returns_number_of_images(self, sample_image_folder):
        ds = ImageFolderDataset(root=str(sample_image_folder))
        assert len(ds) == 5  # 3 cat + 2 dog

    def test_class_names_are_sorted(self, sample_image_folder):
        ds = ImageFolderDataset(root=str(sample_image_folder))
        assert ds.classes == ["cat", "dog"]

    def test_getitem_returns_tuple(self, sample_image_folder):
        ds = ImageFolderDataset(root=str(sample_image_folder))
        img, label = ds[0]
        assert isinstance(label, int)
        assert 0 <= label < 2

    def test_invalid_root_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            ImageFolderDataset(root="/nonexistent/path")

    def test_empty_folder_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(RuntimeError, match="No images found"):
            ImageFolderDataset(root=str(empty_dir))

    def test_samples_property(self, sample_image_folder):
        ds = ImageFolderDataset(root=str(sample_image_folder))
        samples = ds.samples
        assert len(samples) == 5
        # Each sample is (path, label)
        for path, label in samples:
            assert isinstance(path, str)
            assert isinstance(label, int)
