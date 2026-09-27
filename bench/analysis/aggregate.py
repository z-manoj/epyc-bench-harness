"""Aggregation into experiment_summary.parquet: validity classification, cross-rep stats, goodput."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from bench.analysis import consistency
from bench.core import status as S
from bench.core.config import ExperimentConfig, expand_points, load_resolved
from bench.core.io import write_json
from bench.runner.storage import rep_history
from bench.system.samples import finalize_samples

AGG_METRICS = [
    "output_tok_s", "total_tok_s", "req_s", "output_tok_s_per_core", "steady_output_tok_s",
    *[f"{m}_{s}_ms" for m in ("ttft", "tpot", "itl", "e2e") for s in ("mean", "p50", "p90", "p95", "p99")],
    "per_user_tok_s_mean", "per_user_tok_s_p50",
    "cpu_server_util_mean", "cpu_server_util_p50", "cpu_server_util_p95",
    "cpu_server_user_mean", "cpu_server_system_mean", "cpu_outside_util_mean", "freq_mean_mhz",
    "rss_peak_mb", "rss_mean_mb", "pss_peak_mb", "rss_delta_vs_baseline_mb", "sys_mem_used_peak_mb",
    "recovery_s", "window_s",
]


def precision_class(p: str) -> str:
    return "BF16" if p.lower() in ("bf16", "bfloat16") else "8-bit"


def load_experiment_config(exp_dir: Path) -> ExperimentConfig:
    return load_resolved(exp_dir / "experiment_config.resolved.yaml")


def config_status(cdir: Path) -> dict[str, Any]:
    p = cdir / "config_status.json"
    return json.loads(p.read_text()) if p.exists() else {"status": S.NOT_RUN}


def analyze(exp_dir: Path) -> pd.DataFrame:
    cfg = load_experiment_config(exp_dir)
    rows = []
    for sc in cfg.server_configs:
        cdir = exp_dir / sc.id
        if (cdir / "samples").exists():
            finalize_samples(cdir)
        cstat = config_status(cdir)
        for p in expand_points(cfg, sc):
            pdir = cdir / p.point_id
            hist = rep_history(pdir)
            d = consistency.decide_point(hist, cfg.run_policy, cfg.thresholds.consistency)
            cls = d.get("classification", S.NOT_RUN) if d["action"] == "done" else S.NOT_RUN
            if hist:
                summary = {
                    "server_config_id": sc.id, **p.as_dict(), "classification": cls, "decision": d,
                    "reps": [{"rep": r["rep"], "status": r["status"], "flags": r.get("flags", []),
                              "status_reasons": r.get("status_reasons", [])} for r in hist],
                }
                write_json(pdir / "point_summary.json", summary)
            selected = set(d.get("selected_reps") or [])
            sel = [r for r in hist if r["rep"] in selected]
            row: dict[str, Any] = {
                "server_config_id": sc.id, "engine": sc.engine, "precision": sc.precision,
                "precision_class": precision_class(sc.precision), "topology": sc.topology,
                "accel": sc.accel, "accel_group": sc.accel_group,
                "n_instances": len(sc.instances), "n_server_cores": len(sc.all_cores),
                "config_status": cstat.get("status"),
                **{k: v for k, v in p.as_dict().items()},
                "prompt_size": str(p.prompt_size),
                "classification": cls,
                "n_reps": len(hist), "n_success": d.get("n_success", 0), "n_failed": d.get("n_failed", 0),
                "selected_reps": json.dumps(sorted(selected)),
                "outlier_reps": json.dumps(d.get("outliers", [])),
                "failed_reps": json.dumps(d.get("failed_reps", [])),
                "flags": json.dumps(sorted({f for r in hist for f in r.get("flags", [])})),
                "max_cv": (d.get("consistency") or {}).get("max_cv"),
            }
            cons = (d.get("consistency") or {}).get("metrics", {})
            for k in AGG_METRICS:
                vals = [r.get("metrics", {}).get(k) for r in sel]
                sp = consistency.metric_spread(vals)
                row[k] = sp["mean"]
                row[f"{k}_std"] = sp["std"]
                row[f"{k}_cv"] = cons.get(k, {}).get("cv", sp["cv"])
            for k, c in cons.items():
                row[f"{k}_cv_pass"] = c.get("passed")
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = add_goodput(df, cfg)
    df.to_parquet(exp_dir / "experiment_summary.parquet", index=False)
    df.to_csv(exp_dir / "experiment_summary.csv", index=False)
    return df


def add_goodput(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Goodput per (server config, prompt size, output len): highest VALID concurrency meeting the SLA."""
    df = df.copy()
    df["meets_sla"] = ((df["ttft_p95_ms"] <= cfg.sla.ttft_p95_ms) & (df["tpot_p95_ms"] <= cfg.sla.tpot_p95_ms)
                       & df["classification"].isin(list(S.VALID_CLASSES)))
    df["is_goodput"] = False
    df["goodput_concurrency"] = np.nan
    df["goodput_tok_s"] = np.nan
    om = df[df["scenario"] == "online_multi"]
    for _, g in om.groupby(["server_config_id", "prompt_size", "output_len"]):
        ok = g[g["meets_sla"]]
        idx = g.index
        if ok.empty:
            continue
        best = ok.loc[ok["concurrency"].idxmax()]
        df.loc[idx, "goodput_concurrency"] = best["concurrency"]
        df.loc[idx, "goodput_tok_s"] = best["output_tok_s"]
        df.loc[best.name, "is_goodput"] = True
    return df


def load_summary(exp_dir: Path, refresh: bool = False) -> pd.DataFrame:
    p = exp_dir / "experiment_summary.parquet"
    if refresh or not p.exists():
        return analyze(exp_dir)
    return pd.read_parquet(p)


def rep_rows(exp_dir: Path, cfg: Optional[ExperimentConfig] = None) -> pd.DataFrame:
    """One row per rep (all statuses) for listing failed / flagged reps."""
    cfg = cfg or load_experiment_config(exp_dir)
    rows = []
    for sc in cfg.server_configs:
        for p in expand_points(cfg, sc):
            pdir = exp_dir / sc.id / p.point_id
            ps = pdir / "point_summary.json"
            outl = set()
            if ps.exists():
                outl = set(json.loads(ps.read_text()).get("decision", {}).get("outliers", []))
            for r in rep_history(pdir):
                m = r.get("metrics", {})
                flags = list(r.get("flags", [])) + ([S.OUTLIER] if r["rep"] in outl else [])
                rows.append({"server_config_id": sc.id, "accel": sc.accel, "point_id": p.point_id, "rep": r["rep"],
                             "status": r["status"], "flags": ",".join(flags),
                             "reasons": " | ".join(r.get("status_reasons", [])),
                             "output_tok_s": m.get("output_tok_s"), "ttft_p50_ms": m.get("ttft_p50_ms"),
                             "tpot_p50_ms": m.get("tpot_p50_ms")})
    return pd.DataFrame(rows)
