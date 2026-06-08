"""Tests for cvnets.research.tracker.ExperimentTracker."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from cvnets.research.tracker import ExperimentTracker


class TestExperimentTracker:

    def test_init_creates_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(base_dir=tmpdir)
            run_dir = tracker.start(run_name="test_run")
            assert os.path.isdir(run_dir)
            assert run_dir.startswith(tmpdir)

    def test_log_metrics_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(base_dir=tmpdir)
            tracker.start(run_name="test")
            tracker.log_metrics({"accuracy": 0.95, "loss": 0.1})
            metrics_path = os.path.join(tracker.run_dir, "metrics.json")
            assert os.path.isfile(metrics_path)
            with open(metrics_path, "r") as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["accuracy"] == 0.95

    def test_log_config_saves_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(base_dir=tmpdir)
            tracker.start(run_name="test")
            config = {"model": {"name": "demo", "layers": []}}
            tracker.log_config(config)
            config_path = os.path.join(tracker.run_dir, "config.yaml")
            assert os.path.isfile(config_path)

    def test_log_artifact_copies_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(base_dir=tmpdir)
            tracker.start(run_name="test")
            src_file = os.path.join(tmpdir, "dummy.txt")
            with open(src_file, "w") as f:
                f.write("hello")
            tracker.log_artifact(src_file)
            artifact_path = os.path.join(tracker.run_dir, "artifacts", "dummy.txt")
            assert os.path.isfile(artifact_path)

    def test_start_without_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(base_dir=tmpdir)
            run_dir = tracker.start()
            assert os.path.isdir(run_dir)
            basename = os.path.basename(run_dir)
            assert len(basename) > 0

    def test_log_multiple_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(base_dir=tmpdir)
            tracker.start(run_name="test")
            tracker.log_metrics({"epoch": 1, "loss": 0.5})
            tracker.log_metrics({"epoch": 2, "loss": 0.3})
            metrics_path = os.path.join(tracker.run_dir, "metrics.json")
            with open(metrics_path, "r") as f:
                data = json.load(f)
            assert len(data) == 2

    def test_finish_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(base_dir=tmpdir)
            tracker.start(run_name="test")
            tracker.finish()
            summary_path = os.path.join(tracker.run_dir, "summary.json")
            assert os.path.isfile(summary_path)
            with open(summary_path, "r") as f:
                summary = json.load(f)
            assert "run_name" in summary
            assert "finished_at" in summary
