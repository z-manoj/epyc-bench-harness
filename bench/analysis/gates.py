"""Sample-based statistics: stability gate, measurement CPU / memory stats and cooldown recovery."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from bench.analysis.metrics import _f
from bench.core.config import GateThresholds
from bench.system.samples import Samples


def _series_stats(a: np.ndarray, prefix: str) -> dict[str, Optional[float]]:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f"{prefix}_mean": None, f"{prefix}_p50": None, f"{prefix}_p95": None}
    return {f"{prefix}_mean": _f(a.mean()), f"{prefix}_p50": _f(np.percentile(a, 50)),
            f"{prefix}_p95": _f(np.percentile(a, 95))}


def _nanrange(a: np.ndarray) -> Optional[float]:
    a = a[np.isfinite(a)]
    return _f(a.max() - a.min()) if a.size else None


def _nanmean(a: np.ndarray) -> Optional[float]:
    a = a[np.isfinite(a)]
    return _f(a.mean()) if a.size else None


def gate_stats(s: Samples, server_cores: list[int], other_cores: list[int]) -> dict[str, Any]:
    su = s.util(server_cores)
    ou = s.util(other_cores) if other_cores else np.zeros(s.n)
    su_f = su[np.isfinite(su)]
    rss = s.rss_total()
    return {
        "n_samples": s.n,
        "server_util_mean": _nanmean(su),
        "server_util_std": _f(su_f.std()) if su_f.size else None,
        "other_util_mean": _nanmean(ou),
        "sys_mem_drift_mb": _nanrange(s.mem["mem_used_mb"]),
        "sys_mem_used_mean_mb": _nanmean(s.mem["mem_used_mb"]),
        "rss_drift_mb": _nanrange(rss),
        "rss_mean_mb": _nanmean(rss),
        "missed_frac": _f(s.missed_frac()),
        "freq_mean_mhz": _f(s.mean_freq(server_cores)),
    }


def evaluate_gate(st: dict[str, Any], thr: GateThresholds, ref_freq: Optional[float],
                  attribute_cores: bool = True) -> tuple[bool, list[str]]:
    reasons = []

    def over(key: str, limit: float, label: str) -> None:
        v = st.get(key)
        if v is None:
            reasons.append(f"{label}: no data")
        elif v > limit:
            reasons.append(f"{label} {v:.3g} > {limit:g}")

    if not st.get("n_samples"):
        return False, ["no samples in window"]
    # Attach mode with no declared cores cannot tell server cores from the rest of the machine.
    if attribute_cores:
        over("server_util_mean", thr.max_server_util_mean_pct, "server-core util mean %")
        over("server_util_std", thr.max_server_util_std_pp, "server-core util std pp")
        over("other_util_mean", thr.max_other_util_mean_pct, "other-core util mean %")
    over("sys_mem_drift_mb", thr.max_sys_mem_drift_mb, "system memory drift MB")
    if st.get("rss_drift_mb") is not None:
        over("rss_drift_mb", thr.max_rss_drift_mb, "server RSS drift MB")
    over("missed_frac", thr.max_missed_frac, "sampler missed fraction")
    f = st.get("freq_mean_mhz")
    if attribute_cores and ref_freq and f:
        dev = abs(f - ref_freq) / ref_freq
        if dev > thr.max_freq_dev_frac:
            reasons.append(f"frequency {f:.0f} MHz deviates {dev:.1%} from reference {ref_freq:.0f} MHz")
    return not reasons, reasons


def measurement_cpu_stats(s: Samples, server_cores: list[int], instance_cores: list[list[int]],
                          outside_cores: list[int], baseline_rss: Optional[float]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_samples": s.n, "missed_frac": _f(s.missed_frac())}
    out.update({k.replace("x_", "cpu_server_util_"): v
                for k, v in _series_stats(s.util(server_cores), "x").items()})
    out["cpu_server_user_mean"] = _nanmean(s.util(server_cores, "user"))
    out["cpu_server_system_mean"] = _nanmean(s.util(server_cores, "system"))
    out["cpu_instance_util_mean"] = [_nanmean(s.util(c)) for c in instance_cores]
    out["cpu_outside_util_mean"] = _nanmean(s.util(outside_cores)) if outside_cores else 0.0
    out["freq_mean_mhz"] = _f(s.mean_freq(server_cores))
    rss = s.rss_total()
    fin = rss[np.isfinite(rss)]
    out["rss_peak_mb"] = _f(fin.max()) if fin.size else None
    out["rss_mean_mb"] = _f(fin.mean()) if fin.size else None
    out["rss_delta_vs_baseline_mb"] = (_f(fin.max() - baseline_rss)
                                       if fin.size and baseline_rss is not None else None)
    pss = s.pss_total()
    pss = pss[np.isfinite(pss)]
    out["pss_peak_mb"] = _f(pss.max()) if pss.size else None
    out["sys_mem_used_peak_mb"] = _f(np.nanmax(s.mem["mem_used_mb"])) if s.n else None
    out["sys_mem_available_min_mb"] = _f(np.nanmin(s.mem["mem_available_mb"])) if s.n else None
    if s.n and s.numa_used.size:
        out["numa_used_peak_mb"] = {str(n): _f(v) for n, v in zip(s.numa_nodes, np.nanmax(s.numa_used, axis=0))}
    return out


def recovery_time(s: Samples, server_cores: list[int], t_from: int, threshold: float,
                  smooth_samples: int = 20) -> Optional[float]:
    """Seconds after t_from until the smoothed server-core util is <= threshold (None if never)."""
    sub = s.slice(t_from, int(s.t_ns[-1]) if s.n else t_from)
    if sub.n == 0:
        return None
    u = np.nan_to_num(sub.util(server_cores), nan=0.0)
    k = max(1, min(smooth_samples, sub.n))
    roll = np.convolve(u, np.ones(k) / k, mode="valid")
    idx = np.nonzero(roll <= threshold)[0]
    if idx.size == 0:
        return None
    return float((sub.t_ns[idx[0] + k - 1] - t_from) / 1e9)


