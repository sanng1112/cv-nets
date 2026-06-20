"""StatsCollector — compute statistics from lists of tensors."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import Tensor


class StatsCollector:
    """Static methods that compute summary statistics from tensor lists."""

    @staticmethod
    def compute(tensors: List[Tensor]) -> Dict[str, float]:
        """Compute standard statistics over *tensors*.
        Returns keys: mean, std, min, max, l2_norm, sparsity, dead_neuron_ratio.
        """
        if not tensors:
            return {}

        cat = torch.cat([t.flatten().float() for t in tensors], dim=0)
        result: Dict[str, float] = {
            "mean": cat.mean().item(),
            "std": cat.std(unbiased=False).item(),
            "min": cat.min().item(),
            "max": cat.max().item(),
            "l2_norm": cat.norm(p=2).item(),
            "sparsity": (cat == 0).float().mean().item(),
        }

        dead_counts: List[float] = []
        for t in tensors:
            if t.dim() >= 2:
                shape = t.shape
                reshaped = t.view(shape[0], shape[1], -1)
                dead = (reshaped.abs().sum(dim=(0, 2)) == 0).sum().item()
                total = shape[1]
                dead_counts.append(dead / total if total > 0 else 0.0)

        result["dead_neuron_ratio"] = (
            sum(dead_counts) / len(dead_counts) if dead_counts else 0.0
        )
        return result

    @staticmethod
    def gradient_norm(tensors: List[Tensor]) -> Dict[str, float]:
        """Compute gradient L2 norm."""
        if not tensors:
            return {}
        cat = torch.cat([t.flatten().float() for t in tensors], dim=0)
        return {"grad_l2_norm": cat.norm(p=2).item()}

    @staticmethod
    def histogram(tensors: List[Tensor], bins: int = 20) -> Dict[str, Any]:
        """Compute histogram of all elements in *tensors*.
        Returns: hist_bin_edges (list), hist_counts (list).
        """
        if not tensors:
            return {"hist_bin_edges": [], "hist_counts": []}

        cat = torch.cat([t.flatten().float() for t in tensors], dim=0)
        hist = torch.histc(cat, bins=bins, min=cat.min().item(), max=cat.max().item())
        bin_edges = torch.linspace(cat.min().item(), cat.max().item(), bins + 1)

        return {
            "hist_bin_edges": bin_edges.tolist(),
            "hist_counts": hist.int().tolist(),
        }
