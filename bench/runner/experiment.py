"""Experiment driver: pins the harness, captures the environment and runs each server config."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from bench.core.config import ExperimentConfig, expand_points, save_resolved
from bench.core.io import log
from bench.runner.dataset import Dataset
from bench.runner.lifecycle import ServerConfigRunner
from bench.system.env import capture, check_expected, tunables


class Experiment:
    def __init__(self, cfg: ExperimentConfig, exp_dir: Path) -> None:
        self.cfg = cfg
        self.dir = exp_dir
        numeric = sorted({s for wl in cfg.workloads for s in wl.prompt_sizes if s != "mixed"})
        self.dataset = Dataset(cfg.dataset_dir, numeric)
        self.baseline_tunables: dict[str, Any] = {}

    def validate_dataset(self) -> None:
        for sc in self.cfg.server_configs:
            for p in expand_points(self.cfg, sc):
                self.dataset.split(p.prompt_size)

    async def run(self, only: Optional[list[str]] = None) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            os.sched_setaffinity(0, self.cfg.platform.harness_cores)
        except OSError as e:
            log(f"warning: cannot pin harness to {self.cfg.platform.harness_cores}: {e}")
        self.validate_dataset()
        resolved = self.dir / "experiment_config.resolved.yaml"
        if not resolved.exists():
            save_resolved(self.cfg, resolved)
        env_json = self.dir / "env" / "env.json"
        if env_json.exists():
            snap = json.loads(env_json.read_text())
        else:
            log("capturing environment")
            snap = capture(self.cfg, self.dir / "env")
        self.baseline_tunables = snap["tunables"]
        problems = check_expected(self.cfg, tunables())
        if problems:
            raise SystemExit("platform does not match expectations: " + "; ".join(problems))
        for sc in self.cfg.server_configs:
            if only and sc.id not in only:
                continue
            await ServerConfigRunner(self, sc).run()
