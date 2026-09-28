"""Combined report for smoke_1024_128.sh runs of several models, one results subdirectory per model (MODEL=...).

Adds the CPU / memory profile from profile/<config>.csv (scripts/proc_sampler.py) and events.csv. Writes report.md,
report.html, summary.csv, profile_summary.csv and img/ under --root.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from smoke_report import METRICS, ORDER, color, load, md_table, save, speedups  # noqa: E402

MODELS = {"llama": "Llama 3.1 8B Instruct", "qwen2": "Qwen2 7B Instruct", "mixtral": "Mixtral 8x7B Instruct v0.1"}
FORMAT = {"llamacpp": {"bf16": "GGUF BF16", "q8": "GGUF Q8_0"}, "vllm": {"bf16": "safetensors BF16", "w8a8": "W8A8 INT8"}}
SAMPLE_S = 0.25


def fmt_of(cfg: str) -> str:
    eng, _, prec = cfg.split("-")
    return FORMAT[eng][prec]


def windows(raw: pd.DataFrame, lab: str):
    d = raw[raw.label == lab]
    return list(zip(d.t_start, d.t_first)), list(zip(d.t_first, d.t_end))


def in_windows(t: pd.Series, wins) -> pd.Series:
    # a sample at t covers (t - SAMPLE_S, t]; count it when its midpoint lies in a window
    mid = t - SAMPLE_S / 2
    m = pd.Series(False, index=t.index)
    for a, b in wins:
        m |= (mid >= a) & (mid <= b)
    return m


def read_profile(f: Path) -> tuple[pd.DataFrame, str]:
    # older samples carry PSS (smaps_rollup, perturbs the server); current ones carry RssAnon
    p = pd.read_csv(f)
    col = "anon_gb" if "anon_gb" in p else "pss_gb"
    return p.assign(mem_gb=p[col]), "anon" if col == "anon_gb" else "PSS"


def profile_summary(exp: Path, raw: pd.DataFrame, labels) -> pd.DataFrame:
    ev = pd.read_csv(exp / "events.csv", names=["config", "event", "t"])
    rows = []
    for lab in labels:
        f = exp / "profile" / f"{lab}.csv"
        if not f.exists():
            continue
        p, mem = read_profile(f)
        e = ev[ev.config == lab].set_index("event").t
        pre, dec = windows(raw, lab)
        mp, md = in_windows(p.t, pre), in_windows(p.t, dec)
        node0 = p.node_used_gb.iloc[0]
        meas = p[(p.t >= e.get("ready", p.t.min())) & (p.t <= e.get("done", p.t.max()))]
        rows.append({
            "config": lab, "format": fmt_of(lab),
            "mem": mem, "peak mem GB": p.mem_gb.max(), "mem in run GB": meas.mem_gb.median(), "peak RSS GB": p.rss_gb.max(),
            "node mem +GB": p.node_used_gb.max() - node0,
            "cores prefill": p.proc_cores[mp].mean(), "cores decode": p.proc_cores[md].mean(),
            "busy % prefill": p.cpus_busy_pct[mp].mean(), "busy % decode": p.cpus_busy_pct[md].mean(),
            "cores idle": meas.proc_cores[~(mp | md)].median(),
            "threads": int(p.nthreads.max()), "procs": int(p.nproc.max()),
        })
    return pd.DataFrame(rows)


def fig_timelines(exp: Path, raw: pd.DataFrame, labels, title: str):
    ev = pd.read_csv(exp / "events.csv", names=["config", "event", "t"])
    n = len(labels)
    cols = 2
    fig, axes = plt.subplots((n + 1) // cols, cols, figsize=(14, 2.9 * ((n + 1) // cols)), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, lab in zip(axes.flat, labels):
        f = exp / "profile" / f"{lab}.csv"
        if not f.exists():
            ax.set_title(f"{lab}: no profile", fontsize=9)
            continue
        p, mem = read_profile(f)
        e = ev[ev.config == lab].set_index("event").t
        t0 = e.get("launch", p.t.iloc[0])
        x = p.t - t0
        ax.plot(x, p.proc_cores, color="#d62728", lw=0.8, label="CPU (cores)")
        ax.set_ylabel("cores", fontsize=8, color="#d62728")
        ax2 = ax.twinx()
        ax2.plot(x, p.mem_gb, color="#1f77b4", lw=1.2, label=f"{mem} (GB)")
        ax2.set_ylabel(f"{mem} GB", fontsize=8, color="#1f77b4")
        ax2.set_ylim(bottom=0)
        pre, dec = windows(raw, lab)
        for a, b in pre:
            ax.axvspan(a - t0, b - t0, color="#ff9896", alpha=0.35, lw=0)
        for a, b in dec:
            ax.axvspan(a - t0, b - t0, color="#aec7e8", alpha=0.25, lw=0)
        if "ready" in e:
            ax.axvline(e["ready"] - t0, color="k", lw=0.8, ls="--")
        ax.set_title(f"{lab}  ({fmt_of(lab)})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7)
        ax.set_xlabel("s since launch (dashed: ready; red: prefill; blue: decode)", fontsize=7)
    fig.suptitle(f"{title}: server CPU (cores) and memory (anon RSS, or PSS) from launch to shutdown", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_bars(summ: pd.DataFrame, title: str):
    panels = [("ttft_s", "TTFT (s), lower is better"), ("tpot_ms", "TPOT (ms), lower is better"),
              ("e2e_s", "E2E latency (s), lower is better"), ("decode_tok_s", "Decode tok/s, higher is better")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (m, t) in zip(axes.flat, panels):
        x = range(len(summ))
        ax.bar(x, summ[f"{m}_mean"], yerr=summ[f"{m}_std"], capsize=4, color=[color(c) for c in summ.config])
        for i, v in enumerate(summ[f"{m}_mean"]):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(x), summ.config, rotation=25, ha="right", fontsize=8)
        ax.set_title(t, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{title}: prompt 1024 / gen 128, single stream (mean ± sd; red = ZenDNN/zentorch)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def fig_cross(allsum: pd.DataFrame):
    cfgs = [c for c in ORDER if c in set(allsum.config)]
    models = [m for m in MODELS if m in set(allsum.model)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    w = 0.8 / len(models)
    for ax, (m, t) in zip(axes, [("ttft_s", "TTFT (s)"), ("decode_tok_s", "Decode tok/s"), ("peak RSS GB", "Peak RSS (GB)")]):
        for j, mod in enumerate(models):
            d = allsum[allsum.model == mod].set_index("config").reindex(cfgs)
            col = f"{m}_mean" if f"{m}_mean" in d else m
            ax.bar([i + (j - (len(models) - 1) / 2) * w for i in range(len(cfgs))], d[col], w, label=MODELS[mod])
        ax.set_xticks(range(len(cfgs)), cfgs, rotation=30, ha="right", fontsize=8)
        ax.set_title(t, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("All models, 32 cores, prompt 1024 / gen 128, single stream", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    root = ap.parse_args().root
    (root / "img").mkdir(exist_ok=True)
    imgs, sections, allsum, allprof, allsp, failed = {}, [], [], [], [], []
    meta = ""
    for mod, name in MODELS.items():
        exp = root / mod
        if not (exp / "raw.jsonl").exists():
            continue
        meta = meta or (exp / "meta.txt").read_text()
        raw, summ = load(exp)
        sp = speedups(summ)
        prof = profile_summary(exp, raw, list(summ.config))
        startup = dict(line.split() for line in (exp / "startup_s.txt").read_text().splitlines() if line.strip())
        tried = [line.split("=== ")[1].strip() for line in (exp / "smoke.log").read_text().splitlines() if "=== " in line]
        failed += [(mod, c) for c in tried if c not in set(summ.config)]
        summ.insert(0, "model", mod)
        summ["startup_s"] = summ.config.map(lambda c: float(startup.get(c, "nan")))
        allsum.append(summ.merge(prof[["config", "peak RSS GB"]], on="config", how="left"))
        allprof.append(prof.assign(model=mod))
        allsp.append(sp.assign(model=mod))
        imgs[f"{mod}_bars"] = save(fig_bars(summ, name), root / f"img/{mod}_metrics.png")
        imgs[f"{mod}_prof"] = save(fig_timelines(exp, raw, list(summ.config), name), root / f"img/{mod}_profile.png")
        table = pd.DataFrame({
            "config": summ.config, "format": summ.config.map(fmt_of),
            **{lab: [f"{m:.{p}f} ± {s:.{p}f}" for m, s in zip(summ[f"{k}_mean"], summ[f"{k}_std"])]
               for k, lab, p in METRICS[:5]},
            "startup s": summ.startup_s.round(0),
            "tokens": summ.tokens_ok.map({True: "ok", False: "MISMATCH"}),
        })
        sp_t = sp.assign(**{c: sp[c].map(lambda v: f"{v:.2f}x") for c in ["ttft_speedup", "tpot_speedup", "e2e_speedup"]})
        sp_t.columns = ["pair", "with ZenDNN", "without", "TTFT speedup", "TPOT speedup", "E2E speedup"]
        sections.append((mod, name, table, sp_t, prof.round(2)))

    allsum = pd.concat(allsum)
    allsum.to_csv(root / "summary.csv", index=False)
    pd.concat(allprof).to_csv(root / "profile_summary.csv", index=False)
    pd.concat(allsp).to_csv(root / "speedups.csv", index=False)
    imgs["cross"] = save(fig_cross(allsum), root / "img/all_models.png")

    overview = allsum.assign(format=allsum.config.map(fmt_of))[
        ["model", "config", "format", "ttft_s_mean", "tpot_ms_mean", "decode_tok_s_mean", "e2e_s_mean", "peak RSS GB"]]
    overview.columns = ["model", "config", "format", "TTFT s", "TPOT ms", "decode tok/s", "E2E s", "peak RSS GB"]
    overview = overview.round(2)

    method = (
        "One config at a time, 32 physical cores (CPUs 0-31, 4 CCDs), memory bound to NUMA node 0. llama.cpp serves "
        "the GGUF files (BF16, Q8_0); vLLM 0.28.0 serves the safetensors (BF16) and W8A8 INT8 checkpoints. Each "
        "format runs with ZenDNN (llama.cpp `GGML_ZENDNN=ON` build / vLLM zentorch) and without (plain llama.cpp "
        "build / stock vLLM CPU). Each request: streaming `/v1/completions`, exactly 1024 random prompt token IDs "
        "(fresh per request, prefix caching off), 128 output tokens with `ignore_eos`, temperature 0. One warmup "
        "request, then 3 measured single-request runs. TTFT = time to first token; TPOT = (E2E - TTFT) / 127.")
    prof_note = (
        "CPU and memory are sampled every 0.25 s from server launch to shutdown, summed over every process in the "
        "server's session (vLLM runs an API server plus an engine-core process). <b>cores</b> = CPU time per wall "
        "second (32 = all compute cores busy); <b>busy %</b> = mean busy share of CPUs 0-31 from /proc/stat; "
        "prefill = request start to first token, decode = first token to last. <b>RSS</b> includes llama.cpp's mmap'd "
        "GGUF pages and counts shared libraries once per vLLM process (~1 GB each). <b>mem</b> = which column the "
        "profile carries: <b>anon</b> (RssAnon, heap only: excludes mmap'd weights) or <b>PSS</b> (smaps_rollup). "
        "Configs profiled with PSS ran slower during startup and warmup: reading smaps_rollup takes the server's "
        "mmap lock for a page-table walk (~1.5 s at 140 GB) and slowed a faulting vLLM worker ~11x, so their "
        "startup times and first-request (warmup) TTFTs are inflated (Llama vLLM zentorch BF16 startup 96 s vs 51 s; "
        "Mixtral llama.cpp ZenDNN BF16 first request 846 s vs 86 s). Measured runs were barely affected: a Llama rerun "
        "with the current sampler matched within 5% TTFT and 0.5% TPOT (<code>_validate_llama/</code>). "
        "<b>node mem +GB</b> = rise in NUMA node 0 MemUsed (includes page cache). "
        "<b>cores idle</b> = median cores between requests (spinning OpenMP threads show up here).")
    caveats = (
        "Smoke test: 3 single-request runs per config, no stability gating. Q8_0 (GGUF block INT8 weights) and W8A8 "
        "(INT8 weights + dynamic per-token INT8 activations) are different schemes. Llama and Qwen2 W8A8 are RedHatAI "
        "(SmoothQuant + GPTQ); Mixtral W8A8 was made locally with round-to-nearest (`scripts/quantize_w8a8.py`). "
        "Mixtral GGUFs were converted locally from the HF checkpoint. The client sits on socket 1 next to idle "
        "k3s containers.")
    fail_txt = ", ".join(f"{m}/{c}" for m, c in failed)

    md = ["# Model × format smoke test: prompt 1024 / generate 128, 32 cores\n", "## Setup\n", f"```\n{meta}```\n",
          method + "\n", "## Overview (means)\n", md_table(overview), "", "![all](img/all_models.png)\n"]
    if failed:
        md += [f"**Failed to start or run:** {fail_txt} (see `<model>/smoke.log`, `<model>/logs/`).\n"]
    for mod, name, table, sp_t, prof in sections:
        md += [f"## {name}\n", "### Latency (mean ± sd over 3 runs)\n", md_table(table), "",
               "### ZenDNN speedup (baseline / accelerated)\n", md_table(sp_t) if len(sp_t) else "_no pairs_", "",
               f"![{mod}](img/{mod}_metrics.png)\n", "### CPU and memory profile\n", md_table(prof), "",
               f"![{mod} profile](img/{mod}_profile.png)\n"]
    md += ["## Profile columns\n", prof_note.replace("<b>", "**").replace("</b>", "**") + "\n", "## Caveats\n",
           caveats + "\n"]
    (root / "report.md").write_text("\n".join(md))

    css = ("body{font-family:system-ui,sans-serif;max-width:1300px;margin:2em auto;padding:0 1em;color:#222}"
           "table{border-collapse:collapse;font-size:12.5px;margin:1em 0}th,td{border:1px solid #ccc;padding:3px 7px;"
           "text-align:right}th{background:#f3f3f3}td:first-child,th:first-child{text-align:left}"
           "pre{background:#f7f7f7;padding:.8em;overflow-x:auto;font-size:12px}img{max-width:100%}"
           ".warn{background:#fff3cd;padding:.6em;border:1px solid #e0c46c}")
    img = lambda k: f'<img src="data:image/png;base64,{imgs[k]}">' if k in imgs else ""  # noqa: E731
    body = [f"<h1>Model × format smoke test: prompt 1024 / generate 128, 32 cores</h1><h2>Setup</h2><pre>{meta}</pre>"
            f"<p>{method}</p><h2>Overview (means)</h2>{overview.to_html(index=False)}{img('cross')}"]
    if failed:
        body.append(f'<p class="warn"><b>Failed to start or run:</b> {fail_txt}</p>')
    for mod, name, table, sp_t, prof in sections:
        body.append(f"<h2>{name}</h2><h3>Latency (mean ± sd over 3 runs)</h3>{table.to_html(index=False)}"
                    f"<h3>ZenDNN speedup (baseline / accelerated)</h3>{sp_t.to_html(index=False)}{img(mod + '_bars')}"
                    f"<h3>CPU and memory profile</h3><p>{prof_note}</p>{prof.to_html(index=False)}{img(mod + '_prof')}")
    body.append(f"<h2>Caveats</h2><p>{caveats}</p>")
    (root / "report.html").write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><title>Model smoke 1024/128</title><style>{css}</style>'
        f'</head><body>{"".join(body)}</body></html>')
    print(f"wrote {root}/report.html, report.md, summary.csv, profile_summary.csv, speedups.csv, img/")


if __name__ == "__main__":
    main()
