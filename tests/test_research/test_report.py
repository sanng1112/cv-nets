"""Tests for cvnets.research.report.LayerReport."""

from __future__ import annotations

import json

import torch
from torch import nn

from cvnets.research.probe import LayerProbe
from cvnets.research.report import LayerReport


class TestLayerReport:

    def test_generate_basic(self) -> None:
        module = nn.Linear(4, 2)
        with LayerProbe() as probe:
            probe.attach(module)
            x = torch.randn(3, 4)
            out = module(x)
            out.sum().backward()

        report = LayerReport.generate(name="fc1", layer_type="Linear", probe=probe)
        assert report["name"] == "fc1"
        assert report["layer_type"] == "Linear"
        assert "activations" in report
        assert "gradients" in report
        assert report["num_forward_passes"] == 1

    def test_generate_no_data(self) -> None:
        probe = LayerProbe()
        report = LayerReport.generate(name="empty", layer_type="Conv2d", probe=probe)
        assert report["num_forward_passes"] == 0
        assert report["activations"] is None
        assert report["gradients"] is None

    def test_to_json(self) -> None:
        module = nn.Linear(2, 2)
        with LayerProbe() as probe:
            probe.attach(module)
            module(torch.randn(2, 2))

        report = LayerReport.generate("lin", "Linear", probe=probe)
        json_str = LayerReport.to_json(report)
        parsed = json.loads(json_str)
        assert parsed["name"] == "lin"

    def test_stats_keys_are_present(self) -> None:
        module = nn.ReLU()
        with LayerProbe() as probe:
            probe.attach(module)
            module(torch.randn(8, 16))
            module(torch.randn(8, 16))

        report = LayerReport.generate("act", "ReLU", probe=probe)
        act_stats = report["activations"]
        assert act_stats is not None
        for key in ("mean", "std", "min", "max", "sparsity", "l2_norm"):
            assert key in act_stats
