"""ZenDNN acceleration: pair accel=zendnn configs with their accel=none baseline and compute speedups."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from bench.core import status as S
from bench.core.config import ExperimentConfig, ServerConfig

# (metric, higher_is_better). Speedup is always reported so that > 1.0 means ZenDNN is better.
SPEEDUP_METRICS: list[tuple[str, bool]] = [
    ("output_tok_s", True), ("total_tok_s", True), ("req_s", True), ("output_tok_s_per_core", True),
    ("per_user_tok_s_p50", True),
    ("ttft_p50_ms", False), ("ttft_p95_ms", False), ("tpot_p50_ms", False), ("tpot_p95_ms", False),
    ("e2e_p50_ms", False),
]
INFO_METRICS = ["cpu_server_util_mean", "rss_peak_mb"]
ACCEL_ENV_PREFIXES = ("ZENDNN", "ZENTORCH", "GGML_ZENDNN")


def pair_key(sc: ServerConfig) -> str:
    """Identity of a config apart from the acceleration backend (binary and ZenDNN env are ignored)."""
    if sc.accel_group:
        return f"group:{sc.accel_group}"
    env = {k: v for k, v in sc.env.items() if not k.upper().startswith(ACCEL_ENV_PREFIXES)}
    return json.dumps({"mode": sc.mode, "engine": sc.engine, "precision": sc.precision, "model": sc.model,
                       "cores": [i.cores for i in sc.instances], "args": sc.args, "env": env}, sort_keys=True)


def accel_pairs(cfg: ExperimentConfig) -> list[tuple[ServerConfig, ServerConfig]]:
    """(zendnn, none) pairs. A key with several configs on one side pairs each against each."""
    groups: dict[str, dict[str, list[ServerConfig]]] = {}
    for sc in cfg.server_configs:
        groups.setdefault(pair_key(sc), {"zendnn": [], "none": []})[sc.accel].append(sc)
    return [(z, n) for g in groups.values() for z in g["zendnn"] for n in g["none"]]


def _ratio(num: Any, den: Any) -> float:
    try:
        num, den = float(num), float(den)
    except (TypeError, ValueError):
        return float("nan")
    return num / den if np.isfinite(num) and np.isfinite(den) and den > 0 else float("nan")


def speedups(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """One row per (pair, workload point) with ZenDNN-vs-baseline speedups."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    by_cfg = {sid: g.set_index("point_id") for sid, g in df.groupby("server_config_id")}
    for z, n in accel_pairs(cfg):
        dz, dn = by_cfg.get(z.id), by_cfg.get(n.id)
        if dz is None or dn is None:
            continue
        for pid in dz.index.intersection(dn.index):
            rz, rn = dz.loc[pid], dn.loc[pid]
            both = rz["classification"] in S.VALID_CLASSES and rn["classification"] in S.VALID_CLASSES
            row: dict[str, Any] = {
                "zendnn_config": z.id, "baseline_config": n.id, "engine": z.engine, "precision": z.precision,
                "topology": z.topology, "point_id": pid, "scenario": rz["scenario"],
                "prompt_size": rz["prompt_size"], "output_len": rz["output_len"],
                "concurrency": rz["concurrency"], "zendnn_class": rz["classification"],
                "baseline_class": rn["classification"], "both_valid": bool(both),
            }
            for m, higher in SPEEDUP_METRICS:
                row[f"{m}_zendnn"] = rz.get(m)
                row[f"{m}_baseline"] = rn.get(m)
                row[f"{m}_speedup"] = _ratio(rz.get(m), rn.get(m)) if higher else _ratio(rn.get(m), rz.get(m))
            for m in INFO_METRICS:
                row[f"{m}_zendnn"] = rz.get(m)
                row[f"{m}_baseline"] = rn.get(m)
            rows.append(row)
    return pd.DataFrame(rows)


def _gmean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0]
    return float(np.exp(np.log(s).mean())) if len(s) else float("nan")


def speedup_summary(sp: pd.DataFrame) -> pd.DataFrame:
    """Geometric-mean speedup per (pair, scenario) over points where both sides are VALID."""
    if sp.empty:
        return pd.DataFrame()
    rows = []
    for (zc, bc, scen), g in sp[sp["both_valid"]].groupby(["zendnn_config", "baseline_config", "scenario"]):
        row = {"zendnn_config": zc, "baseline_config": bc, "scenario": scen, "points": len(g)}
        for m, _ in SPEEDUP_METRICS:
            row[f"{m}_speedup_gmean"] = _gmean(g[f"{m}_speedup"])
        row["output_tok_s_speedup_min"] = float(g["output_tok_s_speedup"].min())
        row["output_tok_s_speedup_max"] = float(g["output_tok_s_speedup"].max())
        rows.append(row)
    return pd.DataFrame(rows)
