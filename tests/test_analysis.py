import numpy as np
import pytest

from bench.analysis import consistency, gates, metrics
from bench.core import status as S
from bench.core.config import ConsistencyThresholds, GateThresholds, RunPolicy, ValidationThresholds
from bench.system.sampler import CPU_FIELDS
from bench.system.samples import Samples

MS = 1_000_000


def rec(t_send_ms, ttft_ms, n_tok, tpot_ms, prompt=128, nominal=128, error="", out_len=None):
    t_send = int(t_send_ms * MS)
    first = t_send + int(ttft_ms * MS)
    chunks = [first + int(i * tpot_ms * MS) for i in range(n_tok)]
    return {"request_id": f"r{t_send_ms}", "t_send": t_send, "t_first_token": first if n_tok else 0,
            "t_done": chunks[-1] if chunks else t_send, "chunk_ts": chunks, "completion_tokens": n_tok,
            "prompt_tokens": prompt, "nominal_prompt_tokens": nominal, "error": error,
            "completion_tokens_source": "usage", "output_len": out_len or n_tok}


def test_request_metric_formulas():
    r = [rec(0, 100, 11, 10), rec(0, 200, 11, 20)]
    m = metrics.request_metrics(r, "online_single", 1, n_server_cores=10)
    assert m["ttft_p50_ms"] == pytest.approx(150)
    assert m["ttft_mean_ms"] == pytest.approx(150)
    # TPOT = (t_done - t_first) / (n - 1)
    assert m["tpot_mean_ms"] == pytest.approx(15)
    assert m["itl_p50_ms"] == pytest.approx(15)
    assert m["e2e_p99_ms"] == pytest.approx(200 + 200, rel=0.01)
    window_s = 0.4
    assert m["window_s"] == pytest.approx(window_s)
    assert m["output_tok_s"] == pytest.approx(22 / window_s)
    assert m["total_tok_s"] == pytest.approx((22 + 256) / window_s)
    assert m["req_s"] == pytest.approx(2 / window_s)
    assert m["output_tok_s_per_core"] == pytest.approx(22 / window_s / 10)
    assert m["per_user_tok_s_mean"] == pytest.approx((100 + 50) / 2)


def test_errors_excluded_from_latency():
    r = [rec(0, 100, 11, 10), rec(0, 0, 0, 0, error="HTTP 500")]
    m = metrics.request_metrics(r, "online_single", 1, 1)
    assert m["n_errors"] == 1 and m["error_frac"] == 0.5
    assert m["ttft_p50_ms"] == pytest.approx(100)


def test_steady_window_excludes_ramp_and_drain():
    # concurrency 2, 6 requests back-to-back per stream
    r = []
    for stream in range(2):
        t = stream * 5
        for _ in range(6):
            x = rec(t, 10, 5, 10)
            r.append(x)
            t = x["t_done"] / MS
    s0, s1, steady = metrics.steady_window(r, "online_multi", 2)
    done = sorted(x["t_done"] for x in r)
    assert steady and s0 == done[1] and s1 == max(x["t_send"] for x in r)
    s0, s1, steady = metrics.steady_window(r, "online_single", 1)
    assert not steady and s0 == 0


def test_token_checks():
    vt = ValidationThresholds()
    good = [rec(i, 10, 8, 5, prompt=129, nominal=128) for i in range(100)]
    t = metrics.token_checks(good, 8, vt)
    assert not t["tokens_fail"] and not t["prompt_fail"] and not t["prompt_warn"]
    warn = good[:-1] + [rec(0, 10, 8, 5, prompt=132, nominal=128)]     # 3.1 %
    t = metrics.token_checks(warn, 8, vt)
    assert t["prompt_warn"] and not t["prompt_fail"]
    fail = good[:-1] + [rec(0, 10, 8, 5, prompt=140, nominal=128)]     # 9 %
    assert metrics.token_checks(fail, 8, vt)["prompt_fail"]
    short = good[:98] + [rec(0, 10, 6, 5), rec(0, 10, 6, 5)]            # 98 % exact
    assert metrics.token_checks(short, 8, vt)["tokens_fail"]


def test_drift_check():
    stable = [rec(i * 100, 5, 10, 10) for i in range(40)]
    d = metrics.drift_check(stable, stable[0]["t_send"], stable[-1]["t_done"], 0.05)
    assert d["passed"]
    slowing = [rec(i * 100, 5, 10, 10 if i < 20 else 14) for i in range(40)]
    d = metrics.drift_check(slowing, slowing[0]["t_send"], slowing[-1]["t_done"], 0.05)
    assert not d["passed"] and d["tpot_p50_drift"] > 0.05


def make_samples(n=600, ncpu=8, busy=None, rss=None, mem=None, freq=3000.0, drop=()):
    busy = np.zeros((n, ncpu)) if busy is None else busy
    cpu = {f: np.zeros((n, ncpu)) for f in CPU_FIELDS}
    cpu["user"] = busy.astype(float)
    cpu["idle"] = 100.0 - busy
    idx = np.array([i for i in range(n + len(drop)) if i not in set(drop)])[:n]
    return Samples(
        cpu_ids=list(range(ncpu)), interval_ms=50, t_ns=idx * 50 * MS, sample_idx=idx, lag_ns=np.zeros(n),
        cpu=cpu, freq=np.full((n, ncpu), freq),
        mem={"mem_used_mb": np.full(n, 1000.0) if mem is None else mem, "mem_total_mb": np.full(n, 64000.0),
             "mem_available_mb": np.full(n, 60000.0)},
        numa_used=np.zeros((n, 1)),
        tree={"rss_mb": (np.full((n, 1), 500.0) if rss is None else rss[:, None]),
              "pss_mb": np.full((n, 1), np.nan)},
        numa_nodes=[0])


def test_gate_pass_and_fail():
    thr = GateThresholds()
    server, other = [2, 3, 4, 5], [6, 7]
    st = gates.gate_stats(make_samples(), server, other)
    ok, reasons = gates.evaluate_gate(st, thr, ref_freq=3000.0)
    assert ok, reasons

    busy = np.zeros((600, 8))
    busy[:, 2:6] = 10.0
    ok, reasons = gates.evaluate_gate(gates.gate_stats(make_samples(busy=busy), server, other), thr, None)
    assert not ok and any("server-core util mean" in r for r in reasons)

    busy = np.zeros((600, 8))
    busy[::2, 2:6] = 5.0                                   # mean 2.5 (ok) but std 2.5 pp (fail)
    ok, reasons = gates.evaluate_gate(gates.gate_stats(make_samples(busy=busy), server, other), thr, None)
    assert not ok and any("std" in r for r in reasons)

    rss = np.linspace(500, 700, 600)
    ok, reasons = gates.evaluate_gate(gates.gate_stats(make_samples(rss=rss), server, other), thr, None)
    assert not ok and any("RSS drift" in r for r in reasons)

    ok, reasons = gates.evaluate_gate(gates.gate_stats(make_samples(freq=2700.0), server, other), thr, 3000.0)
    assert not ok and any("frequency" in r for r in reasons)

    s = make_samples(drop=range(100, 110))                 # 10 of 610 deadlines missed
    ok, reasons = gates.evaluate_gate(gates.gate_stats(s, server, other), thr, None)
    assert not ok and any("missed" in r for r in reasons)


def test_recovery_time():
    busy = np.zeros((200, 4))
    busy[:100, :] = 80.0
    s = make_samples(n=200, ncpu=4, busy=busy)
    rt = gates.recovery_time(s, [0, 1, 2, 3], 0, threshold=2.0, smooth_samples=10)
    assert rt == pytest.approx(109 * 0.05)
    assert gates.recovery_time(make_samples(n=50, ncpu=4, busy=np.full((50, 4), 50.0)),
                               [0, 1, 2, 3], 0, 2.0) is None


def reps(values, key="output_tok_s", start=1, status=S.SUCCESS):
    out = []
    for i, v in enumerate(values):
        m = {k: 100.0 for k in ConsistencyThresholds().metric_thresholds()}
        m[key] = v
        out.append({"rep": start + i, "status": status, "metrics": m})
    return out


def test_cv_and_consistency():
    sp = consistency.metric_spread([100, 102, 98])
    assert sp["mean"] == pytest.approx(100)
    assert sp["cv"] == pytest.approx(np.std([100, 102, 98], ddof=1) / 100)
    assert sp["range_pct"] == pytest.approx(4.0)
    thr = ConsistencyThresholds()
    assert consistency.consistency([r["metrics"] for r in reps([100, 101, 99])], thr)["passed"]
    c = consistency.consistency([r["metrics"] for r in reps([100, 120, 80])], thr)
    assert not c["passed"] and not c["metrics"]["output_tok_s"]["passed"]
    # e2e p99 is report-only
    c = consistency.consistency([r["metrics"] for r in reps([100, 150, 60], key="e2e_p99_ms")], thr)
    assert c["passed"] and not c["metrics"]["e2e_p99_ms"]["passed"]


def test_decide_point_classification():
    pol, thr = RunPolicy(), ConsistencyThresholds()
    assert consistency.decide_point([], pol, thr)["action"] == "run"
    d = consistency.decide_point(reps([100, 101, 99]), pol, thr)
    assert d["classification"] == S.VALID and d["selected_reps"] == [1, 2, 3]
    # inconsistent -> needs an extra rep
    assert consistency.decide_point(reps([100, 130, 99]), pol, thr)["action"] == "run"
    # extra reps fix it: reps 3,4,5 are consistent
    d = consistency.decide_point(reps([130, 60, 100, 101, 99]), pol, thr)
    assert d["classification"] == S.VALID_WITH_RERUNS and d["selected_reps"] == [3, 4, 5]
    # early stop as soon as a consistent window exists
    d = consistency.decide_point(reps([130, 100, 101, 99]), pol, thr)
    assert d["classification"] == S.VALID_WITH_RERUNS and d["selected_reps"] == [2, 3, 4]
    # never consistent -> UNSTABLE after max_extra_reps
    d = consistency.decide_point(reps([100, 130, 70, 140, 60]), pol, thr)
    assert d["classification"] == S.UNSTABLE
    # failed reps don't count, and are bounded
    hist = reps([100]) + reps([0], start=2, status=S.FAILED_CRASH) + reps([101, 99], start=3)
    d = consistency.decide_point(hist, pol, thr)
    assert d["classification"] == S.VALID and d["selected_reps"] == [1, 3, 4] and d["n_failed"] == 1
    hist = reps([0, 0, 0, 0], status=S.INVALID_BASELINE)
    assert consistency.decide_point(hist, pol, thr)["classification"] == S.FAILED


def test_outliers():
    thr = ConsistencyThresholds()
    assert consistency.outliers(reps([100, 101, 99]), thr) == []
    assert consistency.outliers(reps([100, 101, 110]), thr) == [3]
