"""Build report.md / report.html / summary.csv (+ PNGs) from a smoke_1024_128.sh results directory."""
import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ORDER = ["llamacpp-zendnn-bf16", "llamacpp-cpu-bf16", "vllm-zentorch-bf16", "vllm-cpu-bf16",
         "llamacpp-zendnn-q8", "llamacpp-cpu-q8", "vllm-zentorch-w8a8", "vllm-cpu-w8a8"]
PAIRS = [("llama.cpp BF16", "llamacpp-zendnn-bf16", "llamacpp-cpu-bf16"),
         ("vLLM BF16", "vllm-zentorch-bf16", "vllm-cpu-bf16"),
         ("llama.cpp Q8_0", "llamacpp-zendnn-q8", "llamacpp-cpu-q8"),
         ("vLLM W8A8", "vllm-zentorch-w8a8", "vllm-cpu-w8a8")]
METRICS = [("ttft_s", "TTFT (s)", 3), ("tpot_ms", "TPOT (ms)", 2), ("e2e_s", "E2E latency (s)", 3),
           ("prefill_tok_s", "Prefill (tok/s)", 1), ("decode_tok_s", "Decode (tok/s)", 2),
           ("out_tok_s", "Output throughput (tok/s)", 2)]
COLORS = {"zendnn": "#d62728", "zentorch": "#d62728", "cpu": "#7f7f7f"}


def color(label):
    return next((c for k, c in COLORS.items() if f"-{k}-" in label), "#1f77b4")


def load(exp: Path):
    df = pd.DataFrame([json.loads(line) for line in (exp / "raw.jsonl").read_text().splitlines() if line.strip()])
    df["out_tok_s"] = df["completion_tokens"] / df["e2e_s"]
    labels = [c for c in ORDER if c in set(df.label)] + sorted(set(df.label) - set(ORDER))
    rows = []
    for lab in labels:
        d = df[df.label == lab]
        row = {"config": lab, "runs": len(d)}
        for m, _, _ in METRICS:
            row[f"{m}_mean"] = d[m].mean()
            row[f"{m}_std"] = d[m].std(ddof=1) if len(d) > 1 else 0.0
            row[f"{m}_cv_pct"] = 100 * row[f"{m}_std"] / row[f"{m}_mean"] if row[f"{m}_mean"] else float("nan")
        row["tokens_ok"] = bool(((d.completion_tokens == d.gen_len) &
                                 (d.prompt_tokens.fillna(d.prompt_len) == d.prompt_len)).all())
        rows.append(row)
    return df, pd.DataFrame(rows)


def speedups(summ):
    s = summ.set_index("config")
    out = []
    for name, acc, base in PAIRS:
        if acc in s.index and base in s.index:
            out.append({
                "pair": name, "accelerated": acc, "baseline": base,
                "ttft_speedup": s.at[base, "ttft_s_mean"] / s.at[acc, "ttft_s_mean"],
                "tpot_speedup": s.at[base, "tpot_ms_mean"] / s.at[acc, "tpot_ms_mean"],
                "e2e_speedup": s.at[base, "e2e_s_mean"] / s.at[acc, "e2e_s_mean"],
            })
    return pd.DataFrame(out)


def fig_bars(summ):
    panels = [("ttft_s", "TTFT (s), lower is better"), ("tpot_ms", "TPOT (ms), lower is better"),
              ("e2e_s", "E2E latency (s), lower is better"), ("out_tok_s", "Output tok/s, higher is better")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (m, title) in zip(axes.flat, panels):
        x = range(len(summ))
        ax.bar(x, summ[f"{m}_mean"], yerr=summ[f"{m}_std"], capsize=4, color=[color(c) for c in summ.config])
        for i, v in enumerate(summ[f"{m}_mean"]):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(x), summ.config, rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Llama 3.1 8B, prompt 1024 / gen 128, single stream (mean ± sd over runs; red = ZenDNN/zentorch)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def fig_speedup(sp):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.25
    for j, (col, lab) in enumerate([("ttft_speedup", "TTFT"), ("tpot_speedup", "TPOT"), ("e2e_speedup", "E2E")]):
        xs = [i + (j - 1) * w for i in range(len(sp))]
        ax.bar(xs, sp[col], w, label=lab)
        for xv, v in zip(xs, sp[col]):
            ax.text(xv, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(range(len(sp)), sp.pair)
    ax.set_ylabel("speedup vs. no ZenDNN (x)")
    ax.set_title("ZenDNN / zentorch speedup over the plain CPU backend", fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_runs(df, summ):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, (m, title) in zip(axes, [("ttft_s", "TTFT per run (s)"), ("tpot_ms", "TPOT per run (ms)")]):
        for lab in summ.config:
            d = df[df.label == lab].sort_values("run")
            ax.plot(d.run, d[m], marker="o", label=lab,
                    linestyle="-" if "bf16" in lab else "--", color=None)
        ax.set_xlabel("run")
        ax.set_xticks(sorted(df.run.unique()))
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def save(fig, path: Path) -> str:
    fig.savefig(path, dpi=110)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, type=Path)
    exp = ap.parse_args().experiment
    df, summ = load(exp)
    sp = speedups(summ)
    summ.to_csv(exp / "summary.csv", index=False)
    df.to_csv(exp / "runs.csv", index=False)
    sp.to_csv(exp / "speedups.csv", index=False)
    (exp / "img").mkdir(exist_ok=True)

    imgs = {"bars": save(fig_bars(summ), exp / "img/metrics.png"),
            "runs": save(fig_runs(df, summ), exp / "img/runs.png")}
    if len(sp):
        imgs["speedup"] = save(fig_speedup(sp), exp / "img/speedup.png")

    meta = (exp / "meta.txt").read_text() if (exp / "meta.txt").exists() else ""
    startup = (exp / "startup_s.txt").read_text() if (exp / "startup_s.txt").exists() else ""
    backend = (exp / "backend_check.txt").read_text() if (exp / "backend_check.txt").exists() else ""
    cmds = "\n".join(f"{p.stem}: {p.read_text().strip()}" for p in sorted((exp / "logs").glob("*.cmd")))

    table = pd.DataFrame({
        "config": summ.config,
        "runs": summ.runs,
        **{lab: [f"{m:.{p}f} ± {s:.{p}f}" for m, s in zip(summ[f"{k}_mean"], summ[f"{k}_std"])]
           for k, lab, p in METRICS},
        "max CV %": summ[[f"{k}_cv_pct" for k, _, _ in METRICS[:3]]].max(axis=1).round(2),
        "tokens 1024/128": summ.tokens_ok.map({True: "ok", False: "MISMATCH"}),
    })
    sp_table = sp.assign(**{c: sp[c].map(lambda v: f"{v:.2f}x") for c in ["ttft_speedup", "tpot_speedup", "e2e_speedup"]})
    sp_table.columns = ["pair", "with ZenDNN", "without", "TTFT speedup", "TPOT speedup", "E2E speedup"]
    per_run = df[["label", "run", "ttft_s", "tpot_ms", "e2e_s", "prompt_tokens", "completion_tokens"]].round(3)

    method = (
        "Each config runs alone. The server is pinned with `numactl --physcpubind` to the server CPUs and "
        "`--membind` to the local NUMA node, with one compute thread per physical core. vLLM compute threads are "
        "bound via `VLLM_CPU_OMP_THREADS_BIND`; its API frontend may use the SMT siblings of the same cores. The "
        "client runs on other cores. Both engines run with the same preloaded runtime: LLVM OpenMP (libiomp5) and "
        "tcmalloc. Every request is a streaming `/v1/completions` call with exactly 1024 prompt "
        "token IDs (BOS + random ordinary-vocabulary IDs, fresh per request, so no prefix-cache reuse), "
        "`max_tokens=128`, `ignore_eos`, temperature 0. One warmup request, then 3 measured runs of one request each. "
        "TTFT = time to first streamed token; TPOT = (E2E - TTFT) / 127."
    )
    caveats = (
        "Smoke test only: 3 single-request runs per config, no stability gating. llama.cpp BF16 vs vLLM BF16 is the "
        "like-for-like comparison. INT8 is close but not identical across engines: llama.cpp Q8_0 is block-wise INT8 "
        "weights (GGUF), vLLM W8A8 is RedHatAI compressed-tensors INT8 weights with dynamic per-token INT8 activations. "
        "The host also runs k3s/containerd, which cannot be isolated without root."
    )
    md = [f"# Smoke test: Llama 3.1 8B Instruct, prompt 1024 / generate 128\n",
          "## Setup\n", f"```\n{meta}```\n", method + "\n", "## Results (mean ± sd over runs)\n", md_table(table), "",
          "## ZenDNN speedup (baseline / accelerated; > 1 means ZenDNN is faster)\n",
          md_table(sp_table) if len(sp) else "_no pairs_", "",
          "![metrics](img/metrics.png)\n", "![speedup](img/speedup.png)\n" if len(sp) else "",
          "![runs](img/runs.png)\n", "## Per-run data\n", md_table(per_run), "",
          "## Server startup (s to /health 200)\n", f"```\n{startup}```\n",
          "## Backend evidence from server logs\n", f"```\n{backend}```\n",
          "## Server commands\n", f"```\n{cmds}\n```\n", "## Caveats\n", caveats + "\n"]
    (exp / "report.md").write_text("\n".join(md))

    css = ("body{font-family:system-ui,sans-serif;max-width:1250px;margin:2em auto;padding:0 1em;color:#222}"
           "table{border-collapse:collapse;font-size:13px;margin:1em 0}th,td{border:1px solid #ccc;padding:4px 8px;"
           "text-align:right}th{background:#f3f3f3}td:first-child,th:first-child{text-align:left}"
           "pre{background:#f7f7f7;padding:.8em;overflow-x:auto;font-size:12px}img{max-width:100%}")
    img = lambda k: f'<img src="data:image/png;base64,{imgs[k]}">' if k in imgs else ""  # noqa: E731
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Smoke 1024/128</title><style>{css}</style></head><body>
<h1>Smoke test: Llama 3.1 8B Instruct, prompt 1024 / generate 128</h1>
<h2>Setup</h2><pre>{meta}</pre><p>{method}</p>
<h2>Results (mean ± sd over runs)</h2>{table.to_html(index=False)}
<h2>ZenDNN speedup</h2><p>baseline / accelerated; &gt; 1 means ZenDNN is faster.</p>{sp_table.to_html(index=False) if len(sp) else ""}
{img("bars")}{img("speedup")}{img("runs")}
<h2>Per-run data</h2>{per_run.to_html(index=False)}
<h2>Server startup (s to /health 200)</h2><pre>{startup}</pre>
<h2>Backend evidence from server logs</h2><pre>{backend}</pre>
<h2>Server commands</h2><pre>{cmds}</pre>
<h2>Caveats</h2><p>{caveats}</p></body></html>"""
    (exp / "report.html").write_text(html)
    print(f"wrote {exp}/report.html, report.md, summary.csv, speedups.csv, runs.csv, img/")


if __name__ == "__main__":
    main()
