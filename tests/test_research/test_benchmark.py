"""Tests for cvnets.research.benchmark.BenchmarkRunner."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from cvnets.research.benchmark import BenchmarkRunner


def make_relu():
    return nn.ReLU(inplace=False)


def make_gelu():
    return nn.GELU()


class TestBenchmarkRunner:

    def test_run_single_variant(self) -> None:
        variants = {"relu": make_relu}
        results = BenchmarkRunner.run(
            variants=variants,
            input_shape=(4, 16),
            num_steps=5,
            num_warmup=1,
        )
        assert "relu" in results
        r = results["relu"]
        assert "forward_time_mean_ms" in r
        assert "forward_time_std_ms" in r
        assert "backward_time_mean_ms" in r
        assert r["num_params"] == 0

    def test_run_returns_comparable(self) -> None:
        variants = {"relu": make_relu, "gelu": make_gelu}
        results = BenchmarkRunner.run(
            variants=variants,
            input_shape=(4, 16),
            num_steps=5,
            num_warmup=1,
        )
        assert set(results.keys()) == {"relu", "gelu"}

    def test_run_with_params(self) -> None:
        variants = {"linear": lambda: nn.Linear(16, 8)}
        results = BenchmarkRunner.run(
            variants=variants,
            input_shape=(4, 16),
            num_steps=5,
            num_warmup=1,
        )
        r = results["linear"]
        assert r["num_params"] == 16 * 8 + 8

    def test_comparison_table(self) -> None:
        variants = {"relu": make_relu, "gelu": make_gelu}
        results = BenchmarkRunner.run(
            variants=variants,
            input_shape=(4, 16),
            num_steps=5,
            num_warmup=1,
        )
        table = BenchmarkRunner.compare(results)
        assert isinstance(table, list)
        assert len(table) == 2
        for row in table:
            assert "variant" in row
            assert "forward_ms" in row

    def test_run_warmup(self) -> None:
        variants = {"relu": make_relu}
        results = BenchmarkRunner.run(
            variants=variants,
            input_shape=(4, 16),
            num_steps=10,
            num_warmup=3,
        )
        assert "relu" in results
