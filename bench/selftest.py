"""`bench selftest`: sampler jitter test and a mock-server end-to-end lifecycle test."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from bench.core import status as S
from bench.core.config import ExperimentConfig, parse_config
from bench.system.samples import iter_frames

REPO = Path(__file__).resolve().parent.parent
MOCK_SERVER = REPO / "tests" / "mock_server.py"


def _harness_and_server_cores() -> tuple[list[int], str]:
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 4:
        raise RuntimeError("selftest needs >= 4 CPUs")
    return cpus[:1], f"{cpus[1]}-{cpus[2]}" if cpus[2] == cpus[1] + 1 else f"{cpus[1]},{cpus[2]}"


def write_synthetic_dataset(root: Path, sizes: list[int], n: int = 12) -> None:
    """Prompts of exactly `size` words (the mock server counts words as tokens)."""
    root.mkdir(parents=True, exist_ok=True)
    for size in sizes:
        with open(root / f"prompts_{size}.jsonl", "w") as f:
            for i in range(n):
                words = [f"w{i}_{j}" for j in range(size)]
                f.write(json.dumps({"id": i, "prompt": " ".join(words)}) + "\n")


def mock_config(root: Path, experiment_id: str = "selftest", port_base: int = 18500,
                mock_args: Optional[list[str]] = None, reps: int = 3, workloads: Optional[list] = None,
                **overrides: Any) -> ExperimentConfig:
    harness, server_cores = _harness_and_server_cores()
    raw: dict[str, Any] = {
        "experiment_id": experiment_id,
        "results_dir": str(root / "results"),
        "dataset_dir": str(root / "data"),
        "dev_mode": True,
        "platform": {"harness_cores": harness, "expected_smt": None, "expected_governor": None,
                     "require_numactl": False},
        "run_policy": {"reps": reps, "max_extra_reps": 2, "stability_window_s": 2, "stability_max_retries": 2,
                       "cooldown_s": 1.5, "sample_interval_ms": 50, "server_ready_timeout_s": 30,
                       "request_timeout_s": 30, "system_check_window_s": 1},
        "warmup": {"server_warmup_requests": 4, "per_rep_warmup_requests": 3, "per_rep_warmup_min_s": 0.5},
        # A shared development machine (often a VM with noisy /proc/cpuinfo MHz) is not idle: relax the
        # checks about other processes and frequency; the gate logic itself is unit-tested.
        "thresholds": {
            "system_check": {"max_mean_util_pct": 100, "max_foreign_proc_cpu_pct": 1000},
            "gate": {"max_other_util_mean_pct": 100, "max_sys_mem_drift_mb": 4096,
                     "max_server_util_mean_pct": 20, "max_server_util_std_pp": 20, "max_freq_dev_frac": 1.0},
            "validation": {"max_outside_util_pct": 100, "max_drift_frac": 0.25},
            "consistency": {"output_tok_s_cv": 0.15, "total_tok_s_cv": 0.15, "ttft_p50_cv": 0.5,
                            "tpot_p50_cv": 0.15, "itl_p50_cv": 0.5, "ttft_p95_cv": 0.5, "tpot_p95_cv": 0.5,
                            "cpu_util_cv": 10, "peak_rss_cv": 0.2},
        },
        "server_configs": [{
            "id": "mock_2c", "engine": "mock", "binary": str(MOCK_SERVER), "model": "mock-model",
            "precision": "bf16", "port_base": port_base,
            "instances": [{"cores": server_cores, "numa_node": 0}],
            "args": ["--parallel", "4", "--tpot-ms", "8", "--ttft-ms", "15", "--close-after-stream",
                     *(mock_args or [])],
        }],
        "workloads": workloads or [
            {"scenario": "online_single", "prompt_sizes": [64], "output_lens": [24], "concurrency": [1],
             "num_requests": 8, "min_measure_s": 1.5},
            {"scenario": "online_multi", "prompt_sizes": [32], "output_lens": [24], "concurrency": [4],
             "num_requests": "max(12, 3*concurrency)", "min_measure_s": 1.5},
        ],
    }
    for k, v in overrides.items():
        raw[k] = v
    return parse_config(raw)


def sampler_test(duration_s: float, cores: list[int]) -> dict[str, Any]:
    tmp = Path(tempfile.mkdtemp(prefix="bench_sampler_"))
    out = tmp / "s.bin"
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "bench.system.sampler", "--out", str(out), "--interval-ms", "50",
                    "--cores", ",".join(map(str, cores)), "--duration-s", str(duration_s)],
                   check=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    final = json.loads(Path(str(out) + ".final.json").read_text())
    batches, _ = iter_frames(out)
    t = np.concatenate([b.column("t_ns").to_numpy() for b in batches])
    idx = np.concatenate([b.column("sample_idx").to_numpy() for b in batches])
    lag = np.concatenate([b.column("lag_ns").to_numpy() for b in batches])
    err_ms = np.abs(np.diff(t) / 1e6 - 50.0)
    expected = int(idx[-1] - idx[0] + 1)
    res = {
        "duration_s": duration_s, "wall_s": time.time() - t0, "samples": int(len(t)),
        "interval_err_p99_ms": float(np.percentile(err_ms, 99)),
        "interval_err_max_ms": float(err_ms.max()),
        "deadline_lag_p99_ms": float(np.percentile(lag, 99) / 1e6),
        "missed_frac": 1.0 - len(t) / expected,
        "cpu_pct_of_core": 100.0 * final["loop_cpu_s"] / final["loop_wall_s"],
    }
    res["passed"] = (res["interval_err_p99_ms"] < 10 and res["missed_frac"] < 0.005
                     and res["cpu_pct_of_core"] < 5.0)
    shutil.rmtree(tmp, ignore_errors=True)
    return res


REQUIRED_CONFIG_FILES = ["server_0.log", "launch.json", "samples.parquet", "events.jsonl",
                         "server_warmup.parquet", "config_status.json"]
REQUIRED_REP_FILES = ["requests.parquet", "gate.json", "summary.json"]


def check_outputs(exp_dir: Path, cfg: ExperimentConfig) -> list[str]:
    from bench.core.config import expand_points
    missing = []
    for rel in ["experiment_config.resolved.yaml", "env/env.json", "experiment_summary.parquet"]:
        if not (exp_dir / rel).exists():
            missing.append(rel)
    for sc in cfg.server_configs:
        cdir = exp_dir / sc.id
        missing += [f"{sc.id}/{f}" for f in REQUIRED_CONFIG_FILES if not (cdir / f).exists()]
        for p in expand_points(cfg, sc):
            pdir = cdir / p.point_id
            if not (pdir / "point_summary.json").exists():
                missing.append(f"{sc.id}/{p.point_id}/point_summary.json")
            reps = sorted(pdir.glob("rep_*"))
            if len(reps) < cfg.run_policy.reps:
                missing.append(f"{sc.id}/{p.point_id}: only {len(reps)} reps")
            for r in reps:
                missing += [f"{r.relative_to(exp_dir)}/{f}" for f in REQUIRED_REP_FILES if not (r / f).exists()]
    return missing


def e2e_test(keep: bool) -> dict[str, Any]:
    from bench.analysis.aggregate import analyze
    from bench.runner.experiment import Experiment
    from bench.runner.storage import rep_history
    from bench.reporting.report import report
    root = Path(tempfile.mkdtemp(prefix="bench_selftest_"))
    write_synthetic_dataset(root / "data", [32, 64])
    cfg = mock_config(root)
    exp_dir = Path(cfg.results_dir) / cfg.experiment_id
    t0 = time.time()
    asyncio.run(Experiment(cfg, exp_dir).run())
    df = analyze(exp_dir)
    rep_out = report(exp_dir, html=True, timelines="all")
    missing = check_outputs(exp_dir, cfg)
    statuses = {}
    for sc in cfg.server_configs:
        for pdir in sorted((exp_dir / sc.id).glob("*_p*")):
            statuses[pdir.name] = [h["status"] for h in rep_history(pdir)]
    res = {
        "elapsed_s": time.time() - t0, "exp_dir": str(exp_dir), "missing": missing, "rep_statuses": statuses,
        "classifications": dict(zip(df["point_id"], df["classification"])) if not df.empty else {},
        "report": str(rep_out),
    }
    all_success = all(st and all(s == S.SUCCESS for s in st) for st in statuses.values())
    res["passed"] = not missing and all(len(v) >= cfg.run_policy.reps for v in statuses.values())
    res["all_reps_success"] = all_success
    if not keep and res["passed"]:
        shutil.rmtree(root, ignore_errors=True)
        res["exp_dir"] += " (removed)"
    return res


def selftest(sampler_s: float = 60.0, skip_sampler: bool = False, keep: bool = False) -> int:
    ok = True
    harness, _ = _harness_and_server_cores()
    if not skip_sampler:
        print(f"== sampler jitter test ({sampler_s:g} s at 50 ms, pinned to {harness})")
        r = sampler_test(sampler_s, harness)
        for k, v in r.items():
            print(f"   {k:24} {v:.4g}" if isinstance(v, float) else f"   {k:24} {v}")
        print("   criteria: interval p99 error < 10 ms, missed < 0.5%, CPU < 5% of one core")
        print(f"   -> {'PASS' if r['passed'] else 'FAIL'}")
        ok &= r["passed"]
    print("== mock-server end-to-end lifecycle test (3 reps per point)")
    r = e2e_test(keep)
    for k in ("elapsed_s", "exp_dir", "rep_statuses", "classifications", "missing", "all_reps_success", "report"):
        print(f"   {k:18} {r[k]}")
    print(f"   -> {'PASS' if r['passed'] else 'FAIL'}")
    ok &= r["passed"]
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1
