"""Experiment report: report.md + self-contained report.html + CSV tables under <exp>/report/."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bench.analysis.acceleration import speedup_summary, speedups
from bench.analysis.aggregate import load_experiment_config, load_summary, rep_rows
from bench.analysis.recommend import recommendations
from bench.core import status as S
from bench.core.config import ExperimentConfig, expand_points
from bench.core.io import load_events
from bench.reporting.doc import Doc, _fmt
from bench.reporting.plots import (_safe, plot_acceleration, plot_batch, plot_online_multi, plot_online_single,
                                   plot_rep_overlay, plot_rep_timeline, rep_phases)
from bench.runner.storage import rep_history
from bench.system.samples import load_samples

POINT_COLUMNS = [
    "server_config_id", "engine", "precision", "accel", "topology", "n_server_cores", "point_id", "scenario",
    "prompt_size", "output_len", "concurrency", "classification", "n_reps", "n_success", "max_cv",
    "output_tok_s", "total_tok_s", "req_s", "output_tok_s_per_core", "ttft_p50_ms", "ttft_p95_ms",
    "tpot_p50_ms", "tpot_p95_ms", "itl_p50_ms", "e2e_p50_ms", "per_user_tok_s_p50", "cpu_server_util_mean",
    "freq_mean_mhz", "rss_peak_mb", "pss_peak_mb", "meets_sla", "goodput_concurrency", "goodput_tok_s", "flags",
]


def deviation_table(df: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(display table, highlight mask, numeric table for CSV) of cross-rep spread of the gating metrics."""
    thr = cfg.thresholds.consistency.metric_thresholds()
    gating = [k for k, (_, g) in thr.items() if g]
    rows, hl, num = [], [], []
    for _, r in df.iterrows():
        row = {"config": r["server_config_id"], "point": r["point_id"], "class": r["classification"]}
        flags = {"config": False, "point": False, "class": r["classification"] not in S.VALID_CLASSES}
        nrow = {"server_config_id": r["server_config_id"], "point_id": r["point_id"],
                "classification": r["classification"]}
        for k in gating:
            mean, std, cv = r.get(k), r.get(f"{k}_std"), r.get(f"{k}_cv")
            ok = mean is not None and mean == mean and cv is not None and cv == cv
            row[k] = f"{_fmt(mean)} ± {_fmt(std)} (cv {cv:.1%})" if ok else _fmt(mean)
            flags[k] = bool(cv is not None and cv == cv and cv > thr[k][0])
            nrow.update({f"{k}_mean": mean, f"{k}_std": std, f"{k}_cv": cv, f"{k}_cv_max": thr[k][0]})
        rows.append(row)
        hl.append(flags)
        num.append(nrow)
    return pd.DataFrame(rows), pd.DataFrame(hl), pd.DataFrame(num)


def report(exp_dir: Path, html: bool = True, timelines: str = "all") -> Path:
    cfg = load_experiment_config(exp_dir)
    df = load_summary(exp_dir, refresh=True)
    out = exp_dir / "report"
    img = out / "img"
    img.mkdir(parents=True, exist_ok=True)
    csv_dir = out / "csv"
    csv_dir.mkdir(exist_ok=True)
    csvs: dict[str, pd.DataFrame] = {}

    doc = Doc(out)
    doc.h(1, f"Benchmark report: {cfg.experiment_id}")
    env_p = exp_dir / "env" / "env.json"
    if env_p.exists():
        env = json.loads(env_p.read_text())
        doc.p(f"Host {env.get('hostname')}, kernel {env.get('kernel')}; tunables: "
              + ", ".join(f"{k}={v}" for k, v in env.get("tunables", {}).items()))
        if env.get("expected_problems"):
            doc.p("Platform expectation problems: " + "; ".join(env["expected_problems"]))
    doc.table(pd.DataFrame([{
        "config": sc.id, "mode": sc.mode, "engine": sc.engine, "precision": sc.precision, "accel": sc.accel,
        "topology": sc.topology, "cores": ";".join(i.cores for i in sc.instances),
        "target": (", ".join(f"{e.host}:{e.port}" for e in sc.endpoints) if sc.mode == "attach"
                   else (sc.binary or "")),
        "args": " ".join(sc.args),
    } for sc in cfg.server_configs]))

    # 1. validity overview
    doc.h(2, "1. Validity overview")
    if not df.empty:
        counts = df.groupby(["server_config_id", "classification"]).size().unstack(fill_value=0).reset_index()
        cst = df.groupby("server_config_id")["config_status"].first().reset_index()
        doc.table(counts.merge(cst, on="server_config_id"))
        csvs["points"] = df[[c for c in POINT_COLUMNS if c in df.columns]]
    reps = rep_rows(exp_dir, cfg)
    if not reps.empty:
        csvs["reps"] = reps
        bad = reps[(reps["status"] != S.SUCCESS) | (reps["flags"] != "")]
        doc.p(f"{len(reps)} reps total; {int((reps['status'] == S.SUCCESS).sum())} SUCCESS. "
              "Failed or flagged reps:")
        doc.table(bad[["server_config_id", "point_id", "rep", "status", "flags", "reasons"]])
    for sc in cfg.server_configs:
        st = exp_dir / sc.id / "config_status.json"
        if st.exists():
            d = json.loads(st.read_text())
            if d.get("status") != S.CONFIG_DONE:
                doc.p(f"Server config {sc.id}: {d.get('status')} - {d.get('detail')}")

    # 2 + 3. timelines and overlays
    doc.h(2, "2. Per-rep timelines and 3. rep overlays")
    doc.p("Timeline panels: per-core CPU heatmap (50 ms, dashed lines delimit server cores), server-core "
          "utilisation with user/system split, server RSS/PSS and system memory, server-core frequency. "
          "Shading: gate (blue), warmup (orange), measure (green), cooldown (purple).")
    for sc in cfg.server_configs:
        cdir = exp_dir / sc.id
        if not cdir.exists():
            continue
        samples = load_samples(cdir)
        events = load_events(cdir / "events.jsonl")
        for p in expand_points(cfg, sc):
            hist = rep_history(cdir / p.point_id)
            if not hist:
                continue
            doc.h(3, f"{sc.id} / {p.point_id}")
            ov = img / _safe(f"overlay_{sc.id}_{p.point_id}.png")
            if plot_rep_overlay(cdir, p.point_id, [h["rep"] for h in hist], events, samples, sc.all_cores, ov):
                doc.img(ov, f"overlay {p.point_id}")
            if samples is None or timelines == "none":
                continue
            for h in hist:
                if timelines == "flagged" and h["status"] == S.SUCCESS and not h.get("flags"):
                    continue
                t0, t1, spans = rep_phases(events, p.point_id, h["rep"])
                if t0 is None or t1 is None:
                    continue
                s = samples.slice(t0, t1)
                if s.n == 0:
                    continue
                path = img / _safe(f"timeline_{sc.id}_{p.point_id}_rep{h['rep']}.png")
                plot_rep_timeline(s, spans, t0, sc.all_cores,
                                  f"{sc.id} {p.point_id} rep {h['rep']}: {h['status']} {' '.join(h.get('flags', []))}",
                                  path)
                doc.img(path, f"timeline rep {h['rep']}")

    # 4. deviation tables
    doc.h(2, "4. Cross-rep deviation (selected reps)")
    if not df.empty:
        disp, hl, num = deviation_table(df, cfg)
        doc.table(disp, hl)
        csvs["deviation"] = num

    # 5. scenario results
    doc.h(2, "5. Scenario results")
    if not df.empty:
        doc.p("Only VALID / VALID_WITH_RERUNS points are plotted for online_single and batch; online_multi "
              "marks valid points with o and invalid with x.")
        for pth in plot_online_single(df, img) + plot_online_multi(df, cfg, img) + plot_batch(df, img):
            doc.img(pth, pth.stem)
        cols = ["server_config_id", "accel", "point_id", "classification", "output_tok_s", "total_tok_s",
                "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms", "per_user_tok_s_p50",
                "output_tok_s_per_core", "cpu_server_util_mean", "rss_peak_mb", "goodput_concurrency",
                "goodput_tok_s"]
        doc.table(df[[c for c in cols if c in df.columns]])

    # 6. ZenDNN acceleration
    doc.h(2, "6. ZenDNN acceleration (accel=zendnn vs accel=none)")
    sp = speedups(df, cfg)
    if sp.empty:
        doc.p("No zendnn/none config pairs with common workload points. Add a matching server config with "
              "accel: none (same engine, precision, model and accel_group or cores/args), or point both "
              "at already-running servers with mode: attach.")
    else:
        doc.p("Speedup > 1.0 means ZenDNN is better: throughput metrics are zendnn / baseline, latency metrics "
              "are baseline / zendnn. Summary uses the geometric mean over points where both sides are VALID.")
        summ = speedup_summary(sp)
        csvs["acceleration"] = sp
        csvs["acceleration_summary"] = summ
        doc.table(summ)
        for pth in plot_acceleration(sp, img):
            doc.img(pth, pth.stem)
        keep = ["zendnn_config", "baseline_config", "point_id", "both_valid", "output_tok_s_baseline",
                "output_tok_s_zendnn", "output_tok_s_speedup", "ttft_p50_ms_baseline", "ttft_p50_ms_zendnn",
                "ttft_p50_ms_speedup", "tpot_p50_ms_baseline", "tpot_p50_ms_zendnn", "tpot_p50_ms_speedup"]
        doc.table(sp[keep], pd.DataFrame({"both_valid": ~sp["both_valid"]}))

    # 7. recommendations
    doc.h(2, "7. Recommendation")
    rec = recommendations(df, cfg) if not df.empty else pd.DataFrame()
    doc.p(recommendations.__doc__.strip().split("\n\n")[1].replace("\n", " "))
    doc.table(rec)
    csvs["recommendations"] = rec

    for name, t in csvs.items():
        t.to_csv(csv_dir / f"{name}.csv", index=False)
    doc.p("CSV tables: " + ", ".join(f"csv/{n}.csv" for n in csvs))
    md = out / "report.md"
    md.write_text(doc.markdown())
    if html:
        (out / "report.html").write_text(doc.html())
    return md
