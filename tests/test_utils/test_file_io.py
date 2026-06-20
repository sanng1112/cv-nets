"""Tests for file_io utilities."""

import pytest
from pathlib import Path

from cvnets.utils import ensure_dir, safe_load_yaml, safe_load_json, save_json


class TestFileIO:
    def test_ensure_dir_creates_directory(self, tmp_path):
        p = tmp_path / "a" / "b" / "c"
        result = ensure_dir(str(p))
        assert result.is_dir()

    def test_safe_load_yaml_valid(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text("model:\n  name: test\n")
        data = safe_load_yaml(str(f))
        assert data == {"model": {"name": "test"}}

    def test_safe_load_yaml_not_found(self):
        with pytest.raises(Exception):
            safe_load_yaml("/nonexistent/file.yaml")

    def test_save_json_and_load(self, tmp_path):
        data = {"key": "value", "num": 42}
        f = tmp_path / "out.json"
        save_json(data, str(f))
        assert f.is_file()
        loaded = safe_load_json(str(f))
        assert loaded == data
