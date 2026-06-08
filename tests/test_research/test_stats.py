"""Tests for cvnets.research.stats.StatsCollector."""

from __future__ import annotations

import pytest
import torch

from cvnets.research.stats import StatsCollector


class TestStatsCollector:

    def test_compute_mean_std(self) -> None:
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([4.0, 5.0, 6.0])
        result = StatsCollector.compute([t1, t2])
        assert "mean" in result
        assert "std" in result
        assert abs(result["mean"] - 3.5) < 1e-6
        expected_std = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).std(unbiased=False).item()
        assert abs(result["std"] - expected_std) < 1e-6

    def test_compute_min_max(self) -> None:
        t1 = torch.tensor([0.5, -1.0, 3.0])
        t2 = torch.tensor([2.0, 7.0, -0.5])
        result = StatsCollector.compute([t1, t2])
        assert abs(result["min"] - (-1.0)) < 1e-6
        assert abs(result["max"] - 7.0) < 1e-6

    def test_compute_norm(self) -> None:
        t = torch.tensor([3.0, 4.0])
        result = StatsCollector.compute([t])
        assert abs(result["l2_norm"] - 5.0) < 1e-6

    def test_compute_sparsity(self) -> None:
        t = torch.tensor([0.0, 1.0, 0.0, 0.0, 2.0])
        result = StatsCollector.compute([t])
        assert abs(result["sparsity"] - 0.6) < 1e-6

    def test_compute_dead_neuron_ratio(self) -> None:
        t = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0]])
        result = StatsCollector.compute([t])
        assert abs(result["dead_neuron_ratio"] - 0.6) < 1e-6

    def test_compute_gradient_norm(self) -> None:
        t = torch.tensor([3.0, 4.0])
        result = StatsCollector.gradient_norm([t])
        assert abs(result["grad_l2_norm"] - 5.0) < 1e-6

    def test_compute_histogram(self) -> None:
        t = torch.tensor([0.0, 0.5, 1.0, 0.25, 0.75])
        result = StatsCollector.histogram([t], bins=4)
        assert "hist_bin_edges" in result
        assert "hist_counts" in result
        assert len(result["hist_bin_edges"]) == 5
        assert len(result["hist_counts"]) == 4
        assert sum(result["hist_counts"]) == 5

    def test_compute_empty_list(self) -> None:
        result = StatsCollector.compute([])
        assert result == {}

    def test_compute_single_tensor(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = StatsCollector.compute([t])
        assert "mean" in result
        assert abs(result["mean"] - 2.5) < 1e-6
