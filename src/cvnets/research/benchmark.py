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

            inp = torch.randn(*input_shape, device=device, requires_grad=True)
            fwd_times: List[float] = []
            bwd_times: List[float] = []

            # Determine whether backward pass is meaningful
            has_params = num_params > 0

            for step in range(num_steps + num_warmup):
                if step > 0:
                    inp = torch.randn(*input_shape, device=device, requires_grad=True)

                t0 = time.perf_counter()
                out = module(inp)
                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                t2 = t1
                t3 = t1
                if has_params or (isinstance(out, torch.Tensor) and out.requires_grad):
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
