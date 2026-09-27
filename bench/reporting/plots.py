"""Matplotlib figures: per-rep timelines, rep overlays, scenario results and ZenDNN speedups."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from bench.core import status as S  # noqa: E402
from bench.core.config import ExperimentConfig  # noqa: E402
from bench.system.samples import Samples  # noqa: E402

PHASE_COLORS = {"gate": "#9ecae1", "warmup": "#fdd0a2", "measure": "#a1d99b", "cooldown": "#dadaeb"}


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


# ---------------------------------------------------------------------------
# Timeline plots


def rep_phases(events: list[dict], point_id: str, rep: int) -> tuple[Optional[int], Optional[int], list]:
    evs = [e for e in events if e.get("point_id") == point_id and e.get("rep") == rep]
    start = next((e["t_ns"] for e in evs if e["phase"] == "rep" and e["event"] == "start"), None)
    end = next((e["t_ns"] for e in reversed(evs) if e["phase"] == "rep" and e["event"] == "end"), None)
    spans, open_ = [], {}
    for e in evs:
        if e["phase"] in PHASE_COLORS:
            if e["event"] == "start":
                open_[e["phase"]] = e["t_ns"]
            elif e["event"] == "end" and e["phase"] in open_:
                spans.append((e["phase"], open_.pop(e["phase"]), e["t_ns"]))
    return start, end, spans


def plot_rep_timeline(s: Samples, spans: list, t0: int, server_cores: list[int], title: str, path: Path) -> None:
    rel = (s.t_ns - t0) / 1e9
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1, 0.8]})
    busy = s.busy().T
    ax = axes[0]
    if busy.size:
        stride = max(1, busy.shape[1] // 3000)
        ext = [rel[0], rel[-1], len(s.cpu_ids) - 0.5, -0.5]
        im = ax.imshow(busy[:, ::stride], aspect="auto", cmap="viridis", vmin=0, vmax=100, extent=ext,
                       interpolation="nearest")
        fig.colorbar(im, ax=ax, pad=0.01, label="busy %")
    ax.set_ylabel("CPU id")
    sc = set(server_cores)
    edges = [i for i, c in enumerate(s.cpu_ids) if (c in sc) != ((s.cpu_ids[i - 1] in sc) if i else False)]
    for e in edges:
        ax.axhline(e - 0.5, color="w", lw=0.6, ls="--")
    ax.set_title(title)

    ax = axes[1]
    ax.stackplot(rel, np.nan_to_num(s.util(server_cores, "user")), np.nan_to_num(s.util(server_cores, "system")),
                 labels=["user", "system"], colors=["#3182bd", "#e6550d"], alpha=0.8)
    ax.plot(rel, s.util(server_cores), color="k", lw=0.5, label="busy")
    ax.set_ylabel("server-core util %")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    ax.plot(rel, s.rss_total() / 1024, label="server RSS (GiB)")
    pss = s.pss_total()
    if np.isfinite(pss).any():
        m = np.isfinite(pss)
        ax.plot(rel[m], pss[m] / 1024, ".", ms=2, label="server PSS (GiB)")
    ax2 = ax.twinx()
    ax2.plot(rel, s.mem["mem_used_mb"] / 1024, color="gray", lw=0.8, label="system used (GiB)")
    ax2.set_ylabel("system used GiB")
    ax.set_ylabel("server GiB")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[3]
    p = s.pos(server_cores)
    if p:
        with np.errstate(all="ignore"):
            ax.plot(rel, np.nanmean(s.freq[:, p], axis=1))
    ax.set_ylabel("MHz (server)")
    ax.set_xlabel("seconds from rep start")
    for a in axes[1:]:
        for ph, a0, a1 in spans:
            a.axvspan((a0 - t0) / 1e9, (a1 - t0) / 1e9, color=PHASE_COLORS[ph], alpha=0.35, lw=0)
    for ph, a0, a1 in spans:
        axes[0].axvline((a0 - t0) / 1e9, color="w", lw=0.8)
        axes[0].text((a0 - t0) / 1e9, -0.5, ph, color="w", fontsize=7, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def throughput_series(req: pd.DataFrame, t0: int, bin_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    m = req[(req["phase"] == "measure") & (req["error"] == "")]
    ts, w = [], []
    for chunks, ct in zip(m["chunk_ts"], m["completion_tokens"]):
        if chunks is None or len(chunks) == 0:
            continue
        a = np.asarray(chunks, dtype=np.int64)
        ts.append(a)
        w.append(np.full(len(a), (ct or len(a)) / len(a)))
    if not ts:
        return np.array([]), np.array([])
    t = (np.concatenate(ts) - t0) / 1e9
    wt = np.concatenate(w)
    edges = np.arange(0, t.max() + bin_s, bin_s)
    h, _ = np.histogram(t, bins=edges, weights=wt)
    width = np.full(len(h), bin_s)
    width[-1] = max(t.max() - edges[-2], 1e-3)   # final bin is only partly covered
    return edges[:-1] + width / 2, h / width


def plot_rep_overlay(cdir: Path, point_id: str, reps: list[int], events: list[dict], samples: Optional[Samples],
                     server_cores: list[int], path: Path) -> bool:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    any_ = False
    for rep in reps:
        rp = cdir / point_id / f"rep_{rep}" / "requests.parquet"
        ms = next((e["t_ns"] for e in events if e.get("point_id") == point_id and e.get("rep") == rep
                   and e["phase"] == "measure" and e["event"] == "start"), None)
        me = next((e["t_ns"] for e in events if e.get("point_id") == point_id and e.get("rep") == rep
                   and e["phase"] == "measure" and e["event"] == "end"), None)
        if not rp.exists() or ms is None:
            continue
        req = pq.read_table(rp).to_pandas()
        x, y = throughput_series(req, ms)
        if len(x):
            axes[0].plot(x, y, label=f"rep {rep}")
            any_ = True
        if samples is not None and me is not None:
            s = samples.slice(ms, me)
            axes[1].plot((s.t_ns - ms) / 1e9, s.util(server_cores), lw=0.7, label=f"rep {rep}")
    axes[0].set_ylabel("output tok/s (1 s bins)")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("server-core util %")
    axes[1].set_xlabel("seconds from measurement start")
    axes[0].set_title(f"{cdir.name} / {point_id}: rep overlay")
    fig.tight_layout()
    if any_:
        fig.savefig(path, dpi=80)
    plt.close(fig)
    return any_


# ---------------------------------------------------------------------------
# Scenario plots


def _label(r: pd.Series) -> str:
    return f"{r['engine']}/{r['precision']}/{r.get('accel', '?')}/{r['topology']} ({r['server_config_id']})"


def _size_key(s: str) -> float:
    return float(s) if s.isdigit() else 1e9


def plot_online_single(df: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    d = df[(df["scenario"] == "online_single") & df["classification"].isin(list(S.VALID_CLASSES))]
    for ol, g in d.groupby("output_len"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for sid, gg in g.groupby("server_config_id"):
            gg = gg.sort_values("prompt_size", key=lambda s: s.map(_size_key))
            lab = _label(gg.iloc[0])
            axes[0].plot(gg["prompt_size"], gg["ttft_p50_ms"], "o-", label=lab)
            axes[1].plot(gg["prompt_size"], gg["per_user_tok_s_p50"], "o-", label=lab)
        axes[0].set(title=f"TTFT p50 vs prompt size (output {ol})", xlabel="prompt tokens", ylabel="ms")
        axes[1].set(title="per-user decode tok/s (1/TPOT p50)", xlabel="prompt tokens", ylabel="tok/s")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        p = out / f"online_single_o{ol}.png"
        fig.savefig(p, dpi=80)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_online_multi(df: pd.DataFrame, cfg: ExperimentConfig, out: Path) -> list[Path]:
    paths = []
    d = df[df["scenario"] == "online_multi"]
    for (ps, ol), g in d.groupby(["prompt_size", "output_len"]):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for sid, gg in g.groupby("server_config_id"):
            gg = gg.sort_values("concurrency")
            valid = gg["classification"].isin(list(S.VALID_CLASSES))
            lab = _label(gg.iloc[0])
            line, = axes[0].plot(gg["concurrency"], gg["output_tok_s"], "-", label=lab)
            col = line.get_color()
            for ax, key in ((axes[0], "output_tok_s"), (axes[1], "ttft_p95_ms"), (axes[2], "tpot_p95_ms")):
                if ax is not axes[0]:
                    ax.plot(gg["concurrency"], gg[key], "-", color=col)
                ax.plot(gg["concurrency"][valid], gg[key][valid], "o", color=col)
                ax.plot(gg["concurrency"][~valid], gg[key][~valid], "x", color=col)
            gp = gg[gg["is_goodput"]]
            if not gp.empty:
                axes[0].plot(gp["concurrency"], gp["output_tok_s"], "*", ms=16, color=col)
        axes[1].axhline(cfg.sla.ttft_p95_ms, color="r", ls="--", lw=1, label="SLA")
        axes[2].axhline(cfg.sla.tpot_p95_ms, color="r", ls="--", lw=1, label="SLA")
        axes[0].set(title=f"throughput (p{ps}, o{ol}); * = goodput", xlabel="concurrency", ylabel="output tok/s")
        axes[1].set(title="TTFT p95", xlabel="concurrency", ylabel="ms")
        axes[2].set(title="TPOT p95", xlabel="concurrency", ylabel="ms")
        for ax in axes:
            ax.set_xscale("log", base=2)
        axes[0].legend(fontsize=7)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        p = out / f"online_multi_p{ps}_o{ol}.png"
        fig.savefig(p, dpi=80)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_batch(df: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    d = df[(df["scenario"] == "batch") & df["classification"].isin(list(S.VALID_CLASSES))]
    for ol, g in d.groupby("output_len"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        sizes = sorted(g["prompt_size"].unique(), key=_size_key)
        x = {s: i for i, s in enumerate(sizes)}
        for sid, gg in g.groupby("server_config_id"):
            gg = gg.sort_values("prompt_size", key=lambda s: s.map(_size_key))
            xs = [x[s] for s in gg["prompt_size"]]
            axes[0].plot(xs, gg["total_tok_s"], "o-", label=_label(gg.iloc[0]))
            axes[1].plot(xs, gg["output_tok_s_per_core"], "o-")
        for ax in axes:
            ax.set_xticks(range(len(sizes)), sizes)
            ax.set_xlabel("prompt tokens")
        axes[0].set(title=f"batch total tok/s (output {ol})", ylabel="tok/s")
        axes[1].set(title="output tok/s per server core", ylabel="tok/s/core")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        p = out / f"batch_o{ol}.png"
        fig.savefig(p, dpi=80)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_acceleration(sp: pd.DataFrame, out: Path) -> list[Path]:
    """Per (zendnn, baseline) pair: grouped bars of throughput / TTFT / TPOT speedup for every point."""
    paths = []
    series = [("output_tok_s_speedup", "output tok/s"), ("ttft_p50_ms_speedup", "TTFT p50"),
              ("tpot_p50_ms_speedup", "TPOT p50")]
    for (zc, bc), g in sp.groupby(["zendnn_config", "baseline_config"]):
        g = g.reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(g) + 3), 4.5))
        x = np.arange(len(g))
        w = 0.8 / len(series)
        for k, (col, lab) in enumerate(series):
            bars = ax.bar(x + (k - (len(series) - 1) / 2) * w, g[col], w, label=lab)
            for b, ok in zip(bars, g["both_valid"]):
                if not ok:
                    b.set_hatch("//")
                    b.set_alpha(0.5)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xticks(x, g["point_id"], rotation=60, ha="right", fontsize=7)
        ax.set(title=f"ZenDNN speedup: {zc} vs {bc} (>1 = ZenDNN better; hatched = not both VALID)",
               ylabel="speedup (x)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out / _safe(f"accel_{zc}_vs_{bc}.png")
        fig.savefig(p, dpi=80)
        plt.close(fig)
        paths.append(p)
    return paths


