"""LayerReport — generate a structured, JSON-serialisable report for a layer."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from cvnets.research.probe import LayerProbe
from cvnets.research.stats import StatsCollector


class LayerReport:

    @staticmethod
    def generate(
        name: str,
        layer_type: str,
        probe: LayerProbe,
        *,
        include_histogram: bool = False,
        histogram_bins: int = 20,
    ) -> Dict[str, Any]:
        num_passes = len(probe.activations)

        act_stats: Optional[Dict[str, Any]] = None
        if probe.activations:
            act_stats = StatsCollector.compute(probe.activations)
            if include_histogram:
                act_stats.update(StatsCollector.histogram(probe.activations, bins=histogram_bins))

        grad_stats: Optional[Dict[str, Any]] = None
        if probe.gradients:
            grad_stats = StatsCollector.compute(probe.gradients)

        return {
            "name": name,
            "layer_type": layer_type,
            "num_forward_passes": num_passes,
            "activations": act_stats,
            "gradients": grad_stats,
        }

    @staticmethod
    def to_json(report: Dict[str, Any], indent: int = 2) -> str:
        return json.dumps(report, indent=indent, default=str)

    @staticmethod
    def print_summary(report: Dict[str, Any]) -> None:
        print(f"\n{'='*60}")
        print(f"  Layer: {report['name']}  ({report['layer_type']})")
        print(f"  Forward passes: {report['num_forward_passes']}")
        print(f"{'='*60}")
        if report["activations"]:
            print("  Activations:")
            for k, v in report["activations"].items():
                if isinstance(v, float):
                    print(f"    {k:>20s}: {v:.4f}")
        if report["gradients"]:
            print("  Gradients:")
            for k, v in report["gradients"].items():
                if isinstance(v, float):
                    print(f"    {k:>20s}: {v:.6f}")
