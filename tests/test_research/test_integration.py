"""Integration tests for the cvnets.research package."""

from __future__ import annotations

import sys
sys.path.insert(0, "src")

import os
import tempfile

import torch

from cvnets.research.probe import LayerProbe
from cvnets.research.stats import StatsCollector
from cvnets.research.report import LayerReport
from cvnets.research.benchmark import BenchmarkRunner
from cvnets.research.tracker import ExperimentTracker


class TestResearchIntegration:

    def test_probe_report_pipeline(self) -> None:
        conv = torch.nn.Conv2d(3, 16, kernel_size=3)
        with LayerProbe() as probe:
            probe.attach(conv)
            x = torch.randn(2, 3, 32, 32)
            out = conv(x)
            out.sum().backward()

        report = LayerReport.generate("conv1", "Conv2d", probe=probe)
        assert report["num_forward_passes"] == 1
        assert report["activations"] is not None
        assert "mean" in report["activations"]
        assert report["gradients"] is not None

    def test_benchmark_activations(self) -> None:
        def make_relu():
            return torch.nn.ReLU()
        def make_gelu():
            return torch.nn.GELU()
        def make_lrelu():
            return torch.nn.LeakyReLU()

        variants = {"relu": make_relu, "gelu": make_gelu, "lrelu": make_lrelu}
        results = BenchmarkRunner.run(
            variants=variants,
            input_shape=(64, 128),
            num_steps=10,
            num_warmup=2,
        )
        assert len(results) == 3
        table = BenchmarkRunner.compare(results)
        assert len(table) == 3

    def test_sparsity_of_relu(self) -> None:
        relu = torch.nn.ReLU()
        with LayerProbe() as probe:
            probe.attach(relu)
            x = torch.randn(32, 64)
            relu(x)

        stats = StatsCollector.compute(probe.activations)
        assert stats["sparsity"] > 0.0

    def test_tracker_full_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(base_dir=tmpdir)
            tracker.start(run_name="integration_test")
            tracker.log_config({"model": {"name": "test"}})
            tracker.log_metrics({"acc": 0.5})
            tracker.log_metrics({"acc": 0.75})
            tracker.finish()

            assert os.path.isfile(os.path.join(tracker.run_dir, "config.yaml"))
            assert os.path.isfile(os.path.join(tracker.run_dir, "metrics.json"))
            assert os.path.isfile(os.path.join(tracker.run_dir, "summary.json"))

    def test_dead_neuron_detection(self) -> None:
        relu = torch.nn.ReLU()
        with LayerProbe() as probe:
            probe.attach(relu)
            x = -torch.ones(4, 8, 2, 2)
            relu(x)

        stats = StatsCollector.compute(probe.activations)
        assert stats["dead_neuron_ratio"] == 1.0
        assert stats["sparsity"] == 1.0
