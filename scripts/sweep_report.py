"""Report for sweep_vllm.sh: vLLM with vs without zentorch across prompt lengths and concurrency."""
import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

STYLE = {"vllm-zentorch-bf16": ("#d62728", "-"), "vllm-cpu-bf16": ("#7f7f7f", "-"),
         "vllm-zentorch-w8a8": ("#ff7f0e", "--"), "vllm-cpu-w8a8": ("#1f77b4", "--")}
PAIRS = [("BF16", "vllm-zentorch-bf16", "vllm-cpu-bf16"), ("W8A8", "vllm-zentorch-w8a8", "vllm-cpu-w8a8")]


def load(exp: Path):
    df = pd.DataFrame([json.loads(x) for x in (exp / "raw.jsonl").read_text().splitlines() if x.strip()])
    runs = (df.groupby(["label", "scenario", "prompt_len", "gen_len", "concurrency", "run"])
              .agg(ttft_s=("ttft_s", "mean"), ttft_max_s=("ttft_s", "max"), tpot_ms=("tpot_ms", "mean"),
                   e2e_s=("e2e_s", "mean"), wall_s=("run_wall_s", "first"),
                   bad=("warning", lambda s: s.notna().sum()) if "warning" in df else ("run", lambda s: 0))
              .reset_index())
    runs["out_tok_s"] = runs.concurrency * runs.gen_len / runs.wall_s
    runs["total_tok_s"] = runs.concurrency * (runs.prompt_len + runs.gen_len) / runs.wall_s
    keys = ["label", "scenario", "prompt_len", "gen_len", "concurrency"]
    metrics = ["ttft_s", "ttft_max_s", "tpot_ms", "e2e_s", "wall_s", "out_tok_s", "total_tok_s"]
    g = runs.groupby(keys)
    summ = g[metrics].mean().add_suffix("_mean").join(g[metrics].std(ddof=1).add_suffix("_std"))
    summ = summ.join(g["run"].count().rename("runs")).join(g["bad"].sum().rename("token_mismatches")).reset_index()
    return df, runs, summ.sort_values(["label", "prompt_len", "concurrency"])


def speedups(summ):
    s = summ.set_index(["label", "scenario"])
    rows = []
    for prec, acc, base in PAIRS:
        for scen in sorted(set(summ[summ.label == acc].scenario) & set(summ[summ.label == base].scenario),
                           key=lambda x: (int(x.split("_c")[1]), int(x[1:].split("_")[0]))):
            a, b = s.loc[(acc, scen)], s.loc[(base, scen)]
            rows.append({"precision": prec, "scenario": scen, "prompt": int(a.prompt_len), "concurrency": int(a.concurrency),
                         "out_tok_s zentorch": a.out_tok_s_mean, "out_tok_s stock": b.out_tok_s_mean,
                         "throughput speedup": a.out_tok_s_mean / b.out_tok_s_mean,
                         "TTFT speedup": b.ttft_s_mean / a.ttft_s_mean,
                         "TPOT speedup": b.tpot_ms_mean / a.tpot_ms_mean})
    return pd.DataFrame(rows)


def fig_concurrency(summ, prompt=1024):
    d = summ[summ.prompt_len == prompt]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (m, title) in zip(axes, [("out_tok_s", "Output throughput (tok/s), higher is better"),
                                     ("ttft_s", "Mean TTFT (s), lower is better"),
                                     ("tpot_ms", "Mean TPOT (ms), lower is better")]):
        for lab, x in d.groupby("label"):
            col, ls = STYLE.get(lab, (None, "-"))
            ax.errorbar(x.concurrency, x[f"{m}_mean"], yerr=x[f"{m}_std"], marker="o", capsize=3,
                        color=col, linestyle=ls, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(d.concurrency.unique()), [str(c) for c in sorted(d.concurrency.unique())])
        ax.set_xlabel("concurrent requests")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"vLLM, prompt {prompt} / gen 128, burst of N simultaneous requests (mean ± sd over 3 runs)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def fig_prompt(summ):
    d = summ[summ.concurrency == 1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (m, title) in zip(axes, [("ttft_s", "TTFT (s), lower is better"), ("tpot_ms", "TPOT (ms), lower is better")]):
        for lab, x in d.groupby("label"):
            col, ls = STYLE.get(lab, (None, "-"))
            ax.errorbar(x.prompt_len, x[f"{m}_mean"], yerr=x[f"{m}_std"], marker="o", capsize=3,
                        color=col, linestyle=ls, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(d.prompt_len.unique()), [str(p) for p in sorted(d.prompt_len.unique())])
        ax.set_xlabel("prompt tokens")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("vLLM, single request, gen 128 (mean ± sd over 3 runs)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def fig_speedup(sp):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    labels = [f"{r.precision}\np{r.prompt} c{r.concurrency}" for r in sp.itertuples()]
    w = 0.27
    for j, (col, lab) in enumerate([("throughput speedup", "output throughput"), ("TTFT speedup", "TTFT"),
                                    ("TPOT speedup", "TPOT")]):
        xs = [i + (j - 1) * w for i in range(len(sp))]
        ax.bar(xs, sp[col], w, label=lab)
        for xv, v in zip(xs, sp[col]):
            ax.text(xv, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(range(len(sp)), labels, fontsize=8)
    ax.set_ylabel("zentorch / stock (x), > 1 = zentorch faster")
    ax.set_title("zentorch speedup over stock vLLM CPU, per scenario", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def save(fig, path):
    fig.savefig(path, dpi=110)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def md_table(df):
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, type=Path)
    exp = ap.parse_args().experiment
    df, runs, summ = load(exp)
    sp = speedups(summ)
    summ.to_csv(exp / "summary.csv", index=False)
    runs.to_csv(exp / "runs.csv", index=False)
    sp.to_csv(exp / "speedups.csv", index=False)
    (exp / "img").mkdir(exist_ok=True)
    imgs = {"conc": save(fig_concurrency(summ), exp / "img/concurrency.png"),
            "prompt": save(fig_prompt(summ), exp / "img/prompt_length.png")}
    if len(sp):
        imgs["speedup"] = save(fig_speedup(sp), exp / "img/speedup.png")

    fmt = lambda m, s, p: f"{m:.{p}f} ± {s:.{p}f}"  # noqa: E731
    table = pd.DataFrame({
        "config": summ.label, "prompt": summ.prompt_len, "concurrency": summ.concurrency, "runs": summ.runs,
        "output tok/s": [fmt(m, s, 1) for m, s in zip(summ.out_tok_s_mean, summ.out_tok_s_std)],
        "total tok/s": [fmt(m, s, 0) for m, s in zip(summ.total_tok_s_mean, summ.total_tok_s_std)],
        "mean TTFT s": [fmt(m, s, 2) for m, s in zip(summ.ttft_s_mean, summ.ttft_s_std)],
        "max TTFT s": [f"{v:.2f}" for v in summ.ttft_max_s_mean],
        "mean TPOT ms": [fmt(m, s, 1) for m, s in zip(summ.tpot_ms_mean, summ.tpot_ms_std)],
        "burst wall s": [fmt(m, s, 2) for m, s in zip(summ.wall_s_mean, summ.wall_s_std)],
        "token mismatches": summ.token_mismatches.astype(int),
    })
    sp_t = sp.copy()
    if len(sp):
        for c in ["out_tok_s zentorch", "out_tok_s stock"]:
            sp_t[c] = sp_t[c].map(lambda v: f"{v:.1f}")
        for c in ["throughput speedup", "TTFT speedup", "TPOT speedup"]:
            sp_t[c] = sp_t[c].map(lambda v: f"{v:.2f}x")

    meta = (exp / "meta.txt").read_text() if (exp / "meta.txt").exists() else ""
    backend = (exp / "backend_check.txt").read_text() if (exp / "backend_check.txt").exists() else ""
    cmds = "\n".join(f"{p.stem}: {p.read_text().strip()}" for p in sorted((exp / "logs").glob("*.cmd")))
    method = ("One vLLM server per (env, model), pinned with numactl to the server CPUs (compute threads via "
              "VLLM_CPU_OMP_THREADS_BIND, frontend on SMT siblings) and memory-bound to the local NUMA node; the client "
              "runs on the other socket. Each scenario: 1 warmup request, then 3 runs; a run is a burst of N "
              "simultaneous streaming /v1/completions requests with exact-length token-ID prompts (fresh random ids "
              "per request, prefix caching off), max_tokens=128, ignore_eos. Output tok/s = N x 128 / burst wall time; "
              "total tok/s also counts prompt tokens. TTFT/TPOT are per-request means within a burst, then averaged "
              "over runs.")

    md = ["# vLLM with vs without zentorch: prompt-length and concurrency sweep\n", "## Setup\n", f"```\n{meta}```\n",
          method + "\n", "## zentorch speedup (zentorch / stock; > 1 means zentorch is faster)\n",
          md_table(sp_t) if len(sp) else "_no pairs_", "", "![speedup](img/speedup.png)\n",
          "![concurrency](img/concurrency.png)\n", "![prompt](img/prompt_length.png)\n",
          "## All results (mean ± sd over runs)\n", md_table(table), "",
          "## Kernel / scheduler evidence from server logs\n", f"```\n{backend}```\n",
          "## Server commands\n", f"```\n{cmds}\n```\n"]
    (exp / "report.md").write_text("\n".join(md))

    css = ("body{font-family:system-ui,sans-serif;max-width:1300px;margin:2em auto;padding:0 1em;color:#222}"
           "table{border-collapse:collapse;font-size:13px;margin:1em 0}th,td{border:1px solid #ccc;padding:4px 8px;"
           "text-align:right}th{background:#f3f3f3}pre{background:#f7f7f7;padding:.8em;overflow-x:auto;font-size:12px}"
           "img{max-width:100%}")
    img = lambda k: f'<img src="data:image/png;base64,{imgs[k]}">' if k in imgs else ""  # noqa: E731
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>vLLM zentorch sweep</title><style>{css}</style></head><body>
<h1>vLLM with vs without zentorch: prompt-length and concurrency sweep</h1>
<h2>Setup</h2><pre>{meta}</pre><p>{method}</p>
<h2>zentorch speedup</h2><p>zentorch / stock; &gt; 1 means zentorch is faster.</p>{sp_t.to_html(index=False) if len(sp) else ""}
{img("speedup")}{img("conc")}{img("prompt")}
<h2>All results (mean ± sd over runs)</h2>{table.to_html(index=False)}
<h2>Kernel / scheduler evidence from server logs</h2><pre>{backend}</pre>
<h2>Server commands</h2><pre>{cmds}</pre></body></html>"""
    (exp / "report.html").write_text(html)
    print(f"wrote {exp}/report.html, report.md, summary.csv, speedups.csv, runs.csv, img/")


if __name__ == "__main__":
    main()
