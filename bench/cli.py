"""bench plan | run | resume | analyze | report | selftest"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from bench.core.config import ConfigError, ExperimentConfig, expand, load_config


def _fmt_h(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    return f"{h}h{rem // 60:02d}m"


def plan_estimate(cfg: ExperimentConfig) -> tuple[list[dict], float]:
    rp, wu, ps = cfg.run_policy, cfg.warmup, cfg.plan
    m = expand(cfg)
    rows, total = [], 0.0
    for sc in cfg.server_configs:
        pts = m.points[sc.id]
        if sc.mode == "attach":
            launch = 5.0
        else:
            launch = (rp.system_check_window_s or rp.stability_window_s) + ps.launch_estimate_s.get(sc.engine, 120)
        warm = wu.server_warmup_requests * ps.server_warmup_request_s
        per_point = [(rp.stability_window_s + wu.per_rep_warmup_min_s + p.min_measure_s + rp.cooldown_s) * rp.reps
                     for p in pts]
        t = launch + warm + sum(per_point)
        total += t
        rows.append({"id": sc.id, "engine": sc.engine, "precision": sc.precision, "accel": sc.accel,
                     "topology": sc.topology,
                     "slots": sc.total_slots, "points": len(pts), "reps": len(pts) * rp.reps,
                     "est_s": t, "points_list": pts})
    return rows, total


def cmd_plan(a: argparse.Namespace) -> int:
    cfg = load_config(a.config)
    rows, total = plan_estimate(cfg)
    print(f"experiment {cfg.experiment_id}  -> {Path(cfg.results_dir) / cfg.experiment_id}")
    print(f"{'server config':32} {'engine':9} {'prec':6} {'accel':6} {'topo':8} {'slots':>5} {'points':>6} {'reps':>5} {'est':>8}")
    for r in rows:
        print(f"{r['id']:32} {r['engine']:9} {r['precision']:6} {r['accel']:6} {r['topology']:8} {r['slots']:5d} "
              f"{r['points']:6d} {r['reps']:5d} {_fmt_h(r['est_s']):>8}")
        if a.verbose:
            for p in r["points_list"]:
                print(f"    {p.point_id:40} requests={p.num_requests:<5} min_measure_s={p.min_measure_s:g}")
    print(f"\nestimated wall time (lower bound, excludes extra/failed reps): {_fmt_h(total)}")
    if total > cfg.plan.budget_hours * 3600:
        print(f"WARNING: estimate exceeds budget of {cfg.plan.budget_hours:g} h", file=sys.stderr)
    return 0


def _run_experiment(cfg: ExperimentConfig, exp_dir: Path, only: list[str] | None, make_report: bool = True) -> int:
    from bench.analysis.aggregate import analyze
    from bench.runner.experiment import Experiment
    try:
        asyncio.run(Experiment(cfg, exp_dir).run(only))
    except KeyboardInterrupt:
        print(f"\ninterrupted; continue with: bench resume --experiment {exp_dir}", file=sys.stderr)
        return 130
    if make_report:
        from bench.reporting.report import report
        md = report(exp_dir, html=True, timelines="flagged")
        print(f"report: {md.with_suffix('.html')} (csv: {md.parent / 'csv'})")
    else:
        analyze(exp_dir)
    print(f"done: {exp_dir}")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    cfg = load_config(a.config)
    exp_dir = Path(cfg.results_dir) / cfg.experiment_id
    if (exp_dir / "experiment_config.resolved.yaml").exists():
        print(f"{exp_dir} already exists; use `bench resume --experiment {exp_dir}`", file=sys.stderr)
        return 2
    return _run_experiment(cfg, exp_dir, a.only, not a.no_report)


def cmd_resume(a: argparse.Namespace) -> int:
    from bench.analysis.aggregate import load_experiment_config
    exp_dir = Path(a.experiment).resolve()
    cfg = load_experiment_config(exp_dir)
    return _run_experiment(cfg, exp_dir, a.only, not a.no_report)


def cmd_analyze(a: argparse.Namespace) -> int:
    from bench.analysis.aggregate import analyze
    df = analyze(Path(a.experiment).resolve())
    if df.empty:
        print("no points")
        return 0
    print(df.groupby(["server_config_id", "classification"]).size().unstack(fill_value=0).to_string())
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    from bench.reporting.report import report
    out = report(Path(a.experiment).resolve(), html=not a.no_html, timelines=a.timelines)
    print(f"report: {out}" + ("" if a.no_html else f", {out.with_suffix('.html')}") + f"; csv: {out.parent / 'csv'}")
    return 0


def cmd_selftest(a: argparse.Namespace) -> int:
    from bench.selftest import selftest
    return selftest(sampler_s=a.sampler_seconds, skip_sampler=a.skip_sampler, keep=a.keep)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="print expanded matrix + estimated wall time")
    p.add_argument("--config", required=True)
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("run", help="execute an experiment")
    p.add_argument("--config", required=True)
    p.add_argument("--only", nargs="*", help="restrict to these server config ids")
    p.add_argument("--no-report", action="store_true", help="skip the HTML/CSV report at the end")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("resume", help="continue an interrupted experiment")
    p.add_argument("--experiment", required=True)
    p.add_argument("--only", nargs="*")
    p.add_argument("--no-report", action="store_true", help="skip the HTML/CSV report at the end")
    p.set_defaults(fn=cmd_resume)
    p = sub.add_parser("analyze", help="build experiment_summary.parquet")
    p.add_argument("--experiment", required=True)
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("report", help="report.md + report.html + csv/*.csv")
    p.add_argument("--experiment", required=True)
    p.add_argument("--no-html", action="store_true", help="skip the self-contained report.html")
    p.add_argument("--timelines", choices=["all", "flagged", "none"], default="all",
                   help="per-rep timeline plots for all reps, only failed/flagged reps, or none")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("selftest", help="sampler jitter test + mock-server end-to-end test")
    p.add_argument("--sampler-seconds", type=float, default=60.0)
    p.add_argument("--skip-sampler", action="store_true")
    p.add_argument("--keep", action="store_true", help="keep the temporary results directory")
    p.set_defaults(fn=cmd_selftest)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
