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
