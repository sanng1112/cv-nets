# cv-nets Research Layer Inspection & Evaluation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Equip cv-nets with a research-grade layer introspection and evaluation toolkit — forward/backward hooks, weight & gradient statistics, activation visualisation, benchmark harness, and experiment tracking — so researchers can inspect, diagnose, and compare every layer in their models.

**Architecture:** A new `cvnets.research` sub-package sits at the Application layer. It provides (1) **Probe hooks** that attach to any `nn.Module` and record forward activations + backward gradients in-memory, (2) **Statistics collectors** that compute histograms, norms, sparsity, dead-neuron ratios from captured data, (3) **A benchmark harness** that runs standardised forward/backward passes across layer variants and produces comparison tables, and (4) **Experiment logging** that persists runs with metadata, config snapshots, and artifacts to disk. All components are configurable via `ConfigResolver`, testable in isolation, and work with both legacy (`layers/`) and new (`cvnets.layers`) code.

**Tech Stack:** Python 3.10+, PyTorch 2.x, pytest, matplotlib (optional), pyyaml, tqdm.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/cvnets/research/__init__.py` | Package init; public API re-exports |
| `src/cvnets/research/probe.py` | `LayerProbe` — forward/backward hook manager with in-memory buffer |
| `src/cvnets/research/stats.py` | `StatsCollector` — compute mean, std, histograms, sparsity, dead neurons |
| `src/cvnets/research/report.py` | `LayerReport` — structured dict/JSON summary of a layer's probe+stats |
| `src/cvnets/research/benchmark.py` | `BenchmarkRunner` — standardised passes, comparison tables |
| `src/cvnets/research/tracker.py` | `ExperimentTracker` — persist runs with metadata, config, artifacts |
| `tests/test_research/__init__.py` | Empty |
| `tests/test_research/test_probe.py` | Tests for `LayerProbe` |
| `tests/test_research/test_stats.py` | Tests for `StatsCollector` |
| `tests/test_research/test_report.py` | Tests for `LayerReport` |
| `tests/test_research/test_benchmark.py` | Tests for `BenchmarkRunner` |
| `tests/test_research/test_tracker.py` | Tests for `ExperimentTracker` |
| `tests/test_research/test_integration.py` | End-to-end smoke tests |

No existing files are modified. The research sub-package is purely additive.


---

### Task 1: LayerProbe — forward/backward hook manager

**Files:**
- Create: `src/cvnets/research/__init__.py`
- Create: `src/cvnets/research/probe.py`
- Create: `tests/test_research/__init__.py`
- Create: `tests/test_research/test_probe.py`

- [ ] **Step 1: Write the failing tests for LayerProbe**

Create `tests/test_research/__init__.py` (empty file).

Create `tests/test_research/test_probe.py`:

```python
"""Tests for cvnets.research.probe.LayerProbe."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from cvnets.research.probe import LayerProbe


class SimpleModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestLayerProbe:

    @pytest.fixture
    def module(self) -> SimpleModule:
        return SimpleModule()

    def test_attach_detach(self, module: SimpleModule) -> None:
        probe = LayerProbe()
        probe.attach(module.linear)
        x = torch.randn(3, 4)
        out = module(x)
        out.sum().backward()
        assert len(probe.activations) == 1
        assert len(probe.gradients) == 1
        probe.detach()
        probe.clear()
        x2 = torch.randn(3, 4)
        out2 = module(x2)
        (out2.sum()).backward()
        assert len(probe.activations) == 0
        assert len(probe.gradients) == 0

    def test_activations_shape(self, module: SimpleModule) -> None:
        probe = LayerProbe()
        probe.attach(module.linear)
        module(torch.randn(3, 4))
        assert probe.activations[0].shape == (3, 2)

    def test_gradients_shape(self, module: SimpleModule) -> None:
        probe = LayerProbe()
        probe.attach(module.linear)
        x = torch.randn(3, 4)
        out = module(x)
        out.sum().backward()
        assert probe.gradients[0].shape == (3, 2)

    def test_clear(self, module: SimpleModule) -> None:
        probe = LayerProbe()
        probe.attach(module.linear)
        module(torch.randn(3, 4))
        assert len(probe.activations) == 1
        probe.clear()
        assert len(probe.activations) == 0

    def test_multiple_forward_passes(self, module: SimpleModule) -> None:
        probe = LayerProbe()
        probe.attach(module.linear)
        for _ in range(5):
            module(torch.randn(3, 4))
        assert len(probe.activations) == 5

    def test_attach_multiple_layers(self) -> None:
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        probe = LayerProbe()
        probe.attach(model[0])
        probe.attach(model[2])
        model(torch.randn(3, 4))
        assert len(probe.activations) == 2

    def test_context_manager(self) -> None:
        module = nn.Linear(4, 2)
        with LayerProbe() as probe:
            probe.attach(module)
            module(torch.randn(3, 4))
            assert len(probe.activations) == 1
        probe.clear()
        module(torch.randn(3, 4))
        assert len(probe.activations) == 0

    def test_detach_all(self, module: SimpleModule) -> None:
        probe = LayerProbe()
        probe.attach(module.linear)
        module(torch.randn(3, 4))
        assert len(probe.activations) == 1
        probe.detach_all()
        probe.clear()
        module(torch.randn(3, 4))
        assert len(probe.activations) == 0
```

- [ ] **Step 2: Run tests to verify they fail**
```bash
cd "/run/media/sanng/New Volume/Github/cv-nets" && python3 -c "import sys; sys.path.insert(0,'src'); from cvnets.research.probe import LayerProbe" 2>&1
```
Expected: `ModuleNotFoundError: No module named 'cvnets.research'`

- [ ] **Step 3: Write minimal implementation**

Create `src/cvnets/research/__init__.py`:

```python
"""cvnets.research — layer inspection, statistics, and benchmarking tools."""

from cvnets.research.probe import LayerProbe          # noqa: F401
from cvnets.research.stats import StatsCollector      # noqa: F401
from cvnets.research.report import LayerReport        # noqa: F401
from cvnets.research.benchmark import BenchmarkRunner # noqa: F401
from cvnets.research.tracker import ExperimentTracker # noqa: F401

__all__ = [
    "LayerProbe",
    "StatsCollector",
    "LayerReport",
    "BenchmarkRunner",
    "ExperimentTracker",
]
```

Create `src/cvnets/research/probe.py`:

```python
"""LayerProbe — lightweight forward/backward hooks for layer introspection."""

from __future__ import annotations

from typing import Any, List, Optional
import torch
from torch import Tensor, nn


class LayerProbe:
    """Capture forward activations and backward gradients from nn.Module hooks.

    Usage
    -----
    >>> model = nn.Linear(4, 2)
    >>> probe = LayerProbe()
    >>> probe.attach(model)
    >>> x = torch.randn(3, 4)
    >>> out = model(x)
    >>> out.sum().backward()
    >>> print(len(probe.activations))   # 1
    >>> print(len(probe.gradients))     # 1
    >>> probe.clear()
    """

    def __init__(self) -> None:
        self._handles: List[Any] = []
        self.activations: List[Tensor] = []
        self.gradients: List[Tensor] = []

    def attach(self, module: nn.Module) -> None:
        """Register forward and backward hooks on *module*."""
        fwd_handle = module.register_forward_hook(self._forward_hook)
        bwd_handle = module.register_full_backward_hook(self._backward_hook)
        self._handles.append(fwd_handle)
        self._handles.append(bwd_handle)

    def detach(self) -> None:
        """Remove hooks from the most recently attached module."""
        if self._handles:
            self._handles.pop().remove()
        if self._handles:
            self._handles.pop().remove()

    def detach_all(self) -> None:
        """Remove all registered hooks."""
        while self._handles:
            self._handles.pop().remove()

    def clear(self) -> None:
        """Empty all recorded buffers."""
        self.activations.clear()
        self.gradients.clear()

    def _forward_hook(self, module: nn.Module, inp: Any, out: Any) -> None:
        if isinstance(out, Tensor):
            self.activations.append(out.detach().clone())
        elif isinstance(out, (tuple, list)):
            for o in out:
                if isinstance(o, Tensor):
                    self.activations.append(o.detach().clone())

    def _backward_hook(self, module: nn.Module, grad_in: Any, grad_out: Any) -> None:
        if isinstance(grad_out[0], Tensor):
            self.gradients.append(grad_out[0].detach().clone())

    def __enter__(self) -> "LayerProbe":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.detach_all()
```

- [ ] **Step 4: Run tests to verify they pass**
```bash
cd "/run/media/sanng/New Volume/Github/cv-nets" && python3 -m pytest tests/test_research/test_probe.py -v
```
Expected: 8 passed

- [ ] **Step 5: Commit**
```bash
git add src/cvnets/research/__init__.py src/cvnets/research/probe.py tests/test_research/__init__.py tests/test_research/test_probe.py
git commit -m "feat: add LayerProbe for forward/backward hook capture"
```

---

### Task 2: StatsCollector — compute statistics from captured tensors

**Files:**
- Create: `src/cvnets/research/stats.py`
- Create: `tests/test_research/test_stats.py`

- [ ] **Step 1: Write the failing tests for StatsCollector**

Create `tests/test_research/test_stats.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**
```bash
cd "/run/media/sanng/New Volume/Github/cv-nets" && python3 -c "import sys; sys.path.insert(0,'src'); from cvnets.research.stats import StatsCollector" 2>&1
```
Expected: ImportError

- [ ] **Step 3: Write minimal implementation**

Create `src/cvnets/research/stats.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**
```bash
cd "/run/media/sanng/New Volume/Github/cv-nets" && python3 -m pytest tests/test_research/test_stats.py -v
```
Expected: 9 passed

- [ ] **Step 5: Commit**
```bash
git add src/cvnets/research/stats.py tests/test_research/test_stats.py
git commit -m "feat: add StatsCollector for layer activation/gradient statistics"
```

---

### Task 3: LayerReport — structured report from probe + stats

**Files:**
- Create: `src/cvnets/research/report.py`
- Create: `tests/test_research/test_report.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_research/test_report.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**
```bash
cd "/run/media/sanng/New Volume/Github/cv-nets" && python3 -c "import sys; sys.path.insert(0,'src'); from cvnets.research.report import LayerReport" 2>&1
```
Expected: ImportError

- [ ] **Step 3: Write minimal implementation**

Create `src/cvnets/research/report.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**
```bash
cd "/run/media/sanng/New Volume/Github/cv-nets" && python3 -m pytest tests/test_research/test_report.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**
```bash
git add src/cvnets/research/report.py tests/test_research/test_report.py
git commit -m "feat: add LayerReport for structured layer inspection summaries"
```

---

### Task 4: BenchmarkRunner — compare layer variants

**Files:**
- Create: `src/cvnets/research/benchmark.py`
- Create: `tests/test_research/test_benchmark.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_research/test_benchmark.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**
```bash
cd "/run/media/sanng/New Volume/Github/cv-nets" && python3 -c "import sys; sys.path.insert(0,'src'); from cvnets.research.benchmark import BenchmarkRunner" 2>&1
```
Expected: ImportError

- [ ] **Step 3: Write minimal implementation**

Create `src/cvnets/research/benchmark.py`:

```python
"""BenchmarkRunner — standardised forward/backward timing across layer variants."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn


class BenchmarkRunner:

    @staticmethod
    def run(
        variants: Dict[str, Callable[[], nn.Module]],
        input_shape: Tuple[int, ...],
        num_steps: int = 100,
        num_warmup: int = 10,
        device: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        results: Dict[str, Dict[str, Any]] = {}

        for name, factory in variants.items():
            module = factory().to(device)
            module.train()

            num_params = sum(p.numel() for p in module.parameters())

            inp = torch.randn(*input_shape, device=device)
            fwd_times: List[float] = []
            bwd_times: List[float] = []

            for step in range(num_steps + num_warmup):
                if step > 0:
                    inp = torch.randn(*input_shape, device=device)

                t0 = time.perf_counter()
                out = module(inp)
                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                loss = out.sum()
                t2 = time.perf_counter()
                loss.backward()
                if device == "cuda":
                    torch.cuda.synchronize()
                t3 = time.perf_counter()

                if step >= num_warmup:
                    fwd_times.append((t1 - t0) * 1000.0)
                    bwd_times.append((t3 - t2) * 1000.0)

            fwd_t = torch.tensor(fwd_times, dtype=torch.float32)
            bwd_t = torch.tensor(bwd_times, dtype=torch.float32)

            results[name] = {
                "forward_time_mean_ms": fwd_t.mean().item(),
                "forward_time_std_ms": fwd_t.std().item(),
                "backward_time_mean_ms": bwd_t.mean().item(),
                "backward_time_std_ms": bwd_t.std().item(),
                "num_params": num_params,
            }

        return results

    @staticmethod
    def compare(results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for variant, stats in results.items():
            rows.append({
                "variant": variant,
                "forward_ms": round(stats["forward_time_mean_ms"], 4),
                "backward_ms": round(stats["backward_time_mean_ms"], 4),
                "num_params": stats["num_params"],
            })
        return rows

    @staticmethod
    def print_table(results: Dict[str, Dict[str, Any]]) -> None:
        rows = BenchmarkRunner.compare(results)
        header = f"{'Variant':<20s} {'Fwd(ms)':>10s} {'Bwd(ms)':>10s} {'Params':>8s}"
        print("\n" + header)
        print("-" * len(header))
        for row in rows:
            print(f"{row['variant']:<20s} {row['forward_ms']:>10.4f} {row['backward_ms']:>10.4f} {row['num_params']:>8d}")
```

- [ ] **Step 4: Run tests to verify they pass**
```bash
cd "/run/media/sanng/New Volume/Github/cv-nets" && python3 -m pytest tests/test_research/test_benchmark.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**
```bash
git add src/cvnets/research/benchmark.py tests/test_research/test_benchmark.py
git commit -m "feat: add BenchmarkRunner for layer comparison and timing"
```

---

### Task 5: ExperimentTracker — persist runs with metadata and artifacts

**Files:**
- Create: `src/cvnets/research/tracker.py`
- Create: `tests/test_research/test_tracker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_research/test_tracker.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**
```bash
cd "/run/media/sanng/New Volume/Github/cv-nets" && python3 -c "import sys; sys.path.insert(0,'src'); from cvnets.research.tracker import ExperimentTracker" 2>&1
```
Expected: ImportError

- [ ] **Step 3: Write minimal implementation**

Create `src/cvnets/research/tracker.py`:

```python
"""ExperimentTracker — persist research runs with metadata, config, metrics."""

from __future__ import annotations

import datetime
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ExperimentTracker:

    def __init__(self, base_dir: str = "./runs") -> None:
        self._base = Path(base_dir)
        self.run_dir: Path = Path(".")
        self._metrics: List[Dict[str, Any]] = []

    def start(self, run_name: Optional[str] = None) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"{run_name}_{ts}" if run_name else ts
        self.run_dir = self._base / folder_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics = []
        return str(self.run_dir.resolve())

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        self._metrics.append(metrics)
        metrics_path = self.run_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(self._metrics, f, indent=2, default=str)

    def log_config(self, config: Dict[str, Any]) -> None:
        config_path = self.run_dir / "config.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def log_artifact(self, src_path: str) -> None:
        art_dir = self.run_dir / "artifacts"
        art_dir.mkdir(exist_ok=True)
        src = Path(src_path)
        dst = art_dir / src.name
        if src.is_file():
            shutil.copy2(src, dst)
        elif src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    def finish(self) -> None:
        summary = {
            "run_name": self.run_dir.name,
            "finished_at": datetime.datetime.now().isoformat(),
            "num_metrics_entries": len(self._metrics),
        }
        summary_path = self.run_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
```

- [ ] **Step 4: Run tests to verify they pass**
```bash
cd "/run/media/sanng/New Volume/Github/cv-nets" && python3 -m pytest tests/test_research/test_tracker.py -v
```
Expected: 7 passed

- [ ] **Step 5: Commit**
```bash
git add src/cvnets/research/tracker.py tests/test_research/test_tracker.py
git commit -m "feat: add ExperimentTracker for research run persistence"
```

---

### Task 6: Integration smoke tests — probe + report + benchmark + tracker

**Files:**
- Create: `tests/test_research/test_integration.py`

- [ ] **Step 1: Write the integration tests**

Create `tests/test_research/test_integration.py`:

```python
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
```

- [ ] **Step 2: Run integration tests**
```bash
cd "/run/media/sanng/New Volume/Github/cv-nets" && python3 -m pytest tests/test_research/test_integration.py -v
```
Expected: 5 passed

- [ ] **Step 3: Run full research test suite**
```bash
cd "/run/media/sanng/New Volume/Github/cv-nets" && python3 -m pytest tests/test_research/ -v
```
Expected: ~33 passed

- [ ] **Step 4: Commit**
```bash
git add tests/test_research/test_integration.py
git commit -m "test: add research package integration smoke tests"
```

---

## Self-Review

### 1. Spec Coverage

| Requirement | Task(s) |
|------------|---------|
| Layer inspection — forward activations | Task 1 (LayerProbe) |
| Layer inspection — backward gradients | Task 1 (LayerProbe) |
| Weight/activation statistics (mean, std, min, max, norm) | Task 2 (StatsCollector) |
| Sparsity and dead-neuron detection | Task 2 (StatsCollector) |
| Histogram data for visualisation | Task 2 (StatsCollector.histogram) |
| Structured JSON report per layer | Task 3 (LayerReport) |
| Benchmark comparison across variants | Task 4 (BenchmarkRunner) |
| Timing (forward/backward in ms) | Task 4 (BenchmarkRunner) |
| Experiment logging (metrics, config, artifacts) | Task 5 (ExperimentTracker) |
| Integration tests | Task 6 |

### 2. Placeholder Scan

No TBD, TODO, "implement later", or "similar to Task N". Every step contains complete, copy-pasteable code with exact file paths and exact commands.

### 3. Type Consistency

- `LayerProbe.activations: List[Tensor]` — consistent across tests and probe.py
- `LayerProbe.gradients: List[Tensor]` — consistent
- `StatsCollector.compute(tensors: List[Tensor]) -> Dict[str, float]` — consistent
- `LayerReport.generate(name: str, layer_type: str, probe: LayerProbe) -> Dict[str, Any]` — consistent
- `BenchmarkRunner.run(variants: Dict[str, Callable], ...) -> Dict[str, Dict[str, Any]]` — consistent
- `ExperimentTracker.start(run_name: Optional[str]) -> str` — consistent

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-06-08-cvnets-research-layer-inspection.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach do you prefer?**
