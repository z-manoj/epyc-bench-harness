"""Cross-rep consistency (spec Section 7): CVs, rep-window selection, outliers and the per-point
repetition decision."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from bench.analysis.metrics import _f
from bench.core import status as S
from bench.core.config import ConsistencyThresholds, RunPolicy


def metric_spread(values: Sequence[Optional[float]]) -> dict[str, Optional[float]]:
    a = np.array([v for v in values if v is not None], dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "std": None, "cv": None, "min": None, "max": None, "range_pct": None, "n": 0}
    mean = a.mean()
    std = a.std(ddof=1) if a.size > 1 else 0.0
    cv = std / abs(mean) if mean else None
    return {"mean": _f(mean), "std": _f(std), "cv": _f(cv) if cv is not None else None,
            "min": _f(a.min()), "max": _f(a.max()),
            "range_pct": _f((a.max() - a.min()) / abs(mean) * 100) if mean else None, "n": int(a.size)}


def consistency(metrics: Sequence[dict[str, Any]], thr: ConsistencyThresholds) -> dict[str, Any]:
    per: dict[str, Any] = {}
    passed = True
    max_cv = 0.0
    for key, (limit, gating) in thr.metric_thresholds().items():
        sp = metric_spread([m.get(key) for m in metrics])
        cv = sp["cv"]
        ok = cv is None or cv <= limit
        sp.update(threshold=limit, gating=gating, passed=bool(ok))
        per[key] = sp
        if gating:
            passed = passed and ok
            if cv is not None:
                max_cv = max(max_cv, cv)
    return {"metrics": per, "passed": bool(passed), "max_cv": max_cv}


def best_window(reps: Sequence[dict[str, Any]], size: int,
                thr: ConsistencyThresholds) -> tuple[Optional[list[int]], Optional[dict[str, Any]]]:
    """Among consecutive windows of `size` successful reps, prefer passing windows, then lowest max-CV.
    Returns (rep indices, consistency result)."""
    best = None
    for i in range(0, len(reps) - size + 1):
        win = reps[i:i + size]
        c = consistency([r["metrics"] for r in win], thr)
        key = (not c["passed"], c["max_cv"])
        if best is None or key < best[0]:
            best = (key, [r["rep"] for r in win], c)
    if best is None:
        return None, None
    return best[1], best[2]


def outliers(reps: Sequence[dict[str, Any]], thr: ConsistencyThresholds,
             key: str = "output_tok_s") -> list[int]:
    """Reps whose `key` is > outlier_frac from the median of the other reps, or > outlier_std std-devs
    (the std criterion only applies when >= 3 other reps make the std meaningful)."""
    flagged = []
    vals = [(r["rep"], r["metrics"].get(key)) for r in reps if r["metrics"].get(key) is not None]
    for rep, v in vals:
        others = np.array([x for rr, x in vals if rr != rep], dtype=np.float64)
        if others.size == 0:
            continue
        med = float(np.median(others))
        dev = abs(v - med)
        if med and dev > thr.outlier_frac * abs(med):
            flagged.append(rep)
        elif others.size >= 3 and dev > thr.outlier_std * others.std(ddof=1):
            flagged.append(rep)
    return flagged


def decide_point(history: Sequence[dict[str, Any]], policy: RunPolicy,
                 thr: ConsistencyThresholds) -> dict[str, Any]:
    """Pure decision over the rep history of one point (ordered by rep number).

    Returns {"action": "run"} when another rep is needed, otherwise the final classification.
    Deterministic in the history, so `resume` reaches the same result as an uninterrupted run.
    """
    succ = [r for r in history if r["status"] == S.SUCCESS]
    failed = [r for r in history if r["status"] != S.SUCCESS]
    need = policy.reps
    budget_ok = len(failed) <= policy.max_failed_reps
    base = {"n_success": len(succ), "n_failed": len(failed),
            "failed_reps": [{"rep": r["rep"], "status": r["status"]} for r in failed]}

    if len(succ) < need:
        if budget_ok:
            return {"action": "run", **base}
        return {"action": "done", "classification": S.FAILED, **base,
                "reason": f"only {len(succ)} successful reps; failed-rep budget exhausted"}

    first = consistency([r["metrics"] for r in succ[:need]], thr)
    if first["passed"]:
        return {"action": "done", "classification": S.VALID, "selected_reps": [r["rep"] for r in succ[:need]],
                "consistency": first, "outliers": outliers(succ, thr), **base}

    if len(succ) > need:
        sel, c = best_window(succ, need, thr)
        if c and c["passed"]:
            return {"action": "done", "classification": S.VALID_WITH_RERUNS, "selected_reps": sel,
                    "consistency": c, "outliers": outliers(succ, thr), **base}
    if len(succ) < need + policy.max_extra_reps and budget_ok:
        return {"action": "run", **base}
    sel, c = best_window(succ, need, thr)
    return {"action": "done", "classification": S.UNSTABLE, "selected_reps": sel, "consistency": c,
            "outliers": outliers(succ, thr), **base}
