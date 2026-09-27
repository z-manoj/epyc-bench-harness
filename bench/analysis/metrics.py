"""Request-level metrics (spec Section 8), token checks and intra-rep drift (pure functions)."""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from bench.core.config import ValidationThresholds

PCTS = (50, 90, 95, 99)


def _f(x: Any) -> Optional[float]:
    """JSON-safe float (NaN/inf -> None)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def dist(values: Iterable[float], prefix: str) -> dict[str, Optional[float]]:
    a = np.asarray(list(values), dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f"{prefix}_mean": None, **{f"{prefix}_p{p}": None for p in PCTS}}
    out = {f"{prefix}_mean": _f(a.mean())}
    for p, v in zip(PCTS, np.percentile(a, PCTS)):
        out[f"{prefix}_p{p}"] = _f(v)
    return out


# ---------------------------------------------------------------------------
# Request-level metrics


def per_request(records: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    ok = [r for r in records if not r["error"] and r["t_first_token"]]
    ttft = np.array([(r["t_first_token"] - r["t_send"]) / 1e6 for r in ok])
    e2e = np.array([(r["t_done"] - r["t_send"]) / 1e6 for r in ok])
    tpot = np.array([(r["t_done"] - r["t_first_token"]) / 1e6 / (r["completion_tokens"] - 1)
                     if (r["completion_tokens"] or 0) > 1 else np.nan for r in ok])
    itl = np.concatenate([np.diff(np.asarray(r["chunk_ts"], dtype=np.int64)) / 1e6
                          for r in ok if len(r["chunk_ts"]) > 1] or [np.array([])])
    return {"ttft": ttft, "e2e": e2e, "tpot": tpot, "itl": itl}


def tokens_in_window(records: Sequence[dict[str, Any]], t0: int, t1: int) -> float:
    """Output tokens emitted in [t0, t1), attributing completion_tokens evenly over streamed chunks."""
    total = 0.0
    for r in records:
        ts = r["chunk_ts"]
        if r["error"] or not len(ts):
            continue
        ct = r["completion_tokens"] or len(ts)
        a = np.asarray(ts, dtype=np.int64)
        n = int(np.count_nonzero((a >= t0) & (a < t1)))
        total += n * ct / len(ts)
    return total


def steady_window(records: Sequence[dict[str, Any]], scenario: str, concurrency: int) -> tuple[int, int, bool]:
    """(start, end, is_steady).  online_multi/batch exclude ramp-up (first `concurrency` completions)
    and drain (after the last request was sent).  Falls back to the full window if too short."""
    ok = [r for r in records if not r["error"]]
    if not ok:
        return 0, 0, False
    full0 = min(r["t_send"] for r in records)
    full1 = max(r["t_done"] for r in records)
    if scenario == "online_single":
        return full0, full1, False
    done = sorted(r["t_done"] for r in ok)
    if len(done) <= concurrency:
        return full0, full1, False
    start = done[concurrency - 1]
    end = max(r["t_send"] for r in records)
    if end - start < 0.2 * (full1 - full0):
        return full0, full1, False
    return start, end, True


def request_metrics(records: Sequence[dict[str, Any]], scenario: str, concurrency: int,
                    n_server_cores: int) -> dict[str, Any]:
    """Metrics over measurement-phase records (Section 8)."""
    m: dict[str, Any] = {}
    n_total = len(records)
    ok = [r for r in records if not r["error"]]
    m["n_requests"] = n_total
    m["n_ok"] = len(ok)
    m["n_errors"] = n_total - len(ok)
    m["error_frac"] = _f((n_total - len(ok)) / n_total) if n_total else None
    if not records:
        return m
    t0 = min(r["t_send"] for r in records)
    t1 = max(r["t_done"] for r in records)
    win_s = max((t1 - t0) / 1e9, 1e-9)
    m["window_start_ns"], m["window_end_ns"], m["window_s"] = t0, t1, win_s
    out_tok = sum(r["completion_tokens"] or 0 for r in ok)
    in_tok = sum(r["prompt_tokens"] or 0 for r in ok)
    m["output_tokens"], m["prompt_tokens"] = out_tok, in_tok
    m["output_tok_s"] = _f(out_tok / win_s)
    m["total_tok_s"] = _f((out_tok + in_tok) / win_s)
    m["req_s"] = _f(len(ok) / win_s)
    m["output_tok_s_per_core"] = _f(out_tok / win_s / n_server_cores) if n_server_cores else None
    pr = per_request(records)
    m.update(dist(pr["ttft"], "ttft"))
    m.update(dist(pr["tpot"], "tpot"))
    m.update(dist(pr["itl"], "itl"))
    m.update(dist(pr["e2e"], "e2e"))
    for k in list(m):
        if k.startswith(("ttft_", "tpot_", "itl_", "e2e_")) and not k.endswith("_ms"):
            m[k + "_ms"] = m.pop(k)
    with np.errstate(divide="ignore"):
        per_user = 1000.0 / pr["tpot"]
    m.update({k.replace("user_tok_s", "per_user_tok_s"): v for k, v in dist(per_user, "user_tok_s").items()})

    s0, s1, steady = steady_window(records, scenario, concurrency)
    m["steady_start_ns"], m["steady_end_ns"], m["steady_is_steady"] = s0, s1, steady
    if s1 > s0:
        m["steady_output_tok_s"] = _f(tokens_in_window(ok, s0, s1) / ((s1 - s0) / 1e9))
    return m


def token_checks(records: Sequence[dict[str, Any]], output_len: int, vt: ValidationThresholds) -> dict[str, Any]:
    ok = [r for r in records if not r["error"]]
    exact = sum(1 for r in ok if r["completion_tokens"] == output_len)
    complete_frac = exact / len(records) if records else 0.0
    devs, unknown = [], 0
    for r in ok:
        nom, act = r["nominal_prompt_tokens"], r["prompt_tokens"]
        if nom is None or act is None:
            unknown += 1
            continue
        devs.append(abs(act - nom) / nom)
    max_dev = max(devs) if devs else None
    return {
        "complete_frac": _f(complete_frac),
        "completion_sources": sorted({str(r["completion_tokens_source"]) for r in ok}),
        "prompt_dev_max": _f(max_dev),
        "prompt_unknown": unknown,
        "prompt_warn": bool(max_dev is not None and max_dev > vt.prompt_warn_frac),
        "prompt_fail": bool(max_dev is not None and max_dev > vt.prompt_fail_frac),
        "tokens_fail": complete_frac < vt.min_complete_frac,
    }


def drift_check(records: Sequence[dict[str, Any]], t0: int, t1: int, max_frac: float,
                gate_tpot: bool = True) -> dict[str, Any]:
    """Split [t0, t1] into halves; compare output tok/s and TPOT p50 of second vs first half.

    With gate_tpot=False the TPOT drift is still reported but does not fail the check. Batch mode needs
    this: requests are admitted in waves, so the first wave's decode shares steps with the next wave's
    prefill and its TPOT differs from later requests by design.
    """
    ok = [r for r in records if not r["error"]]
    out: dict[str, Any] = {"window": [t0, t1]}
    if t1 <= t0 or not ok:
        out.update(passed=True, skipped="empty window")
        return out
    mid = (t0 + t1) // 2
    half_s = (mid - t0) / 1e9
    tps = [tokens_in_window(ok, t0, mid) / half_s, tokens_in_window(ok, mid, t1) / ((t1 - mid) / 1e9)]
    tpots: list[list[float]] = [[], []]
    for r in ok:
        ct = r["completion_tokens"] or 0
        if ct <= 1:
            continue
        center = (r["t_first_token"] + r["t_done"]) // 2
        if t0 <= center < t1:
            tpots[0 if center < mid else 1].append((r["t_done"] - r["t_first_token"]) / 1e6 / (ct - 1))
    out["tok_s_halves"] = [_f(x) for x in tps]
    out["tok_s_drift"] = _f(abs(tps[1] - tps[0]) / tps[0]) if tps[0] > 0 else None
    passed = out["tok_s_drift"] is not None and out["tok_s_drift"] <= max_frac
    if all(len(t) >= 2 for t in tpots):
        p = [float(np.median(t)) for t in tpots]
        out["tpot_p50_halves"] = p
        out["tpot_p50_drift"] = _f(abs(p[1] - p[0]) / p[0]) if p[0] > 0 else None
        out["tpot_gating"] = gate_tpot
        if gate_tpot:
            passed = passed and out["tpot_p50_drift"] is not None and out["tpot_p50_drift"] <= max_frac
    else:
        out["tpot_p50_drift"] = None
        out["tpot_note"] = "too few requests per half for TPOT drift"
    out["passed"] = bool(passed)
    return out


