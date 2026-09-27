# CPU LLM Serving Benchmark Harness (llama.cpp + vLLM on ZenDNN)

Benchmarks Llama 3.1 8B serving on AMD EPYC: llama.cpp (`llama-server`, ggml-zendnn) and vLLM (zentorch),
scenarios `online_single` / `online_multi` / `batch`, with and without ZenDNN, producing an HTML report and CSVs.

## Commands

```bash
source .venv/bin/activate            # Python 3.12 venv; package installed editable (pip install -e .)
bench plan    --config examples/local_llamacpp_q8.yaml [-v]   # expanded matrix + wall-time estimate
bench run     --config <cfg.yaml> [--only <server_config_id>...] [--no-report]
bench resume  --experiment <results_dir>/<experiment_id>       # deterministic; no duplicate reps
bench analyze --experiment <exp_dir>                           # experiment_summary.parquet/.csv
bench report  --experiment <exp_dir> [--no-html] [--timelines all|flagged|none]
bench selftest [--sampler-seconds 60] [--skip-sampler] [--keep]  # sampler jitter + mock e2e
pytest -q                            # 26 tests; e2e tests are marked `slow` (~3.5 min total)
```

`run`/`resume` end by writing `<exp>/report/report.html`, `report.md`, `csv/*.csv` and `img/*.png`.
Ctrl-C exits 130; `resume` removes incomplete rep dirs and continues.

## Layout

| Package | Role |
|---|---|
| `bench/core/` | `config.py` (pydantic models, `extra="forbid"`, matrix expansion, safe `eval_expr`), `status.py` (all status/flag constants), `io.py` (`log`, atomic `write_json`, `EventLog`) |
| `bench/system/` | `sampler.py` (separate process, 50 ms deadline-scheduled `/proc` sampler, Arrow IPC frames; run as `python -m bench.system.sampler`), `samples.py` (reader, `SampleStore`, `finalize_samples` → `samples.parquet`), `env.py` (tunables, versions, checksums, env capture) |
| `bench/runner/` | `experiment.py` (top level), `lifecycle.py` (`ServerConfigRunner`: system check → launch → warmup → per rep gate/warmup/measure/cooldown/validate), `server.py` (`ServerGroup`: pinning, readiness, backend detection), `loadgen.py` (aiohttp streaming SSE), `dataset.py`, `storage.py` |
| `bench/analysis/` | `metrics.py` (request metrics, token checks, drift), `gates.py` (stability gate, CPU stats), `consistency.py` (cross-rep CV, outliers, `decide_point`), `aggregate.py` (summary table, goodput), `acceleration.py` (ZenDNN vs baseline pairing and speedups), `recommend.py` |
| `bench/reporting/` | `doc.py` (Markdown + self-contained HTML), `plots.py` (matplotlib Agg), `report.py` (sections + CSV export) |
| `bench/selftest.py` | acceptance self-test; `mock_config()` is reused by e2e tests |
| `tests/` | `mock_server.py` (OpenAI-compatible mock: `--backend zendnn|cpu`, `--accel-speedup`, crash/EOS/close-after-stream knobs), `test_config.py`, `test_analysis.py`, `test_e2e.py` |
| `examples/` | `experiment.yaml` (full spec), `local_llamacpp_q8.yaml` (this VM, ZenDNN + CPU baselines, ~40 h), `dev_llamacpp_accel.yaml` (short ZenDNN-vs-CPU run, ~45 min), `dev_llamacpp_smoke.yaml` |
| `prompts/` | `prompts_{128..4096}.txt`, JSONL `{id, text}`, 20 prompts each |

## Conventions

- Every threshold lives in `core/config.py` `Thresholds` and is overridable from YAML; don't hard-code limits.
- `decide_point` must stay a pure function of rep history (resume determinism). Rep summaries are written atomically.
- Status precedence and names come from `core/status.py`; add new statuses there.
- `ServerConfig.accel`: `zendnn` (default) or `none`. Configs pair for speedup analysis when engine, precision,
  model, cores, args and env match, ignoring `binary` and env vars prefixed `ZENDNN`/`ZENTORCH`/`GGML_ZENDNN`,
  or explicitly via a shared `accel_group`.
- Backend verification: llama.cpp uses `llama-server --list-devices` only (the log contains binary paths such as
  `build-nozendnn`); a mismatch aborts the config with `INVALID_BACKEND`. vLLM mismatch only warns.
- Load generator uses `force_close=True` (llama-server closes the connection after each streamed response) and
  sends `cache_prompt: false` to llama.cpp.
- Comments state constraints only; match surrounding style.

## Environment (this VM)

- AMD EPYC 9V74, 8 cores / 16 threads, SMT siblings adjacent (0-1, 2-3, ...), 1 NUMA node, 62 GB RAM.
- No cpufreq (frequency read from `/proc/cpuinfo`, noisy, so `gate.max_freq_dev_frac: 0.25`), no numactl
  (`require_numactl: false`, pinning via `sched_setaffinity`), no sudo. `/` read-only.
- `/home` is the only persistent disk and is ~96% full (~1.4 GB free). `/tmp` is a 58 GB RAM-backed tmpfs,
  wiped on reboot.
- Harness runs on cpus 0-1; servers on 2-15.
- llama.cpp: `/home/mkumar/llama.cpp` (commit 9adc7f4). ZenDNN build `build/bin/llama-server`;
  plain CPU build `build-nozendnn/bin/llama-server` (`-DGGML_ZENDNN=OFF`, lists no devices).
  ggml-zendnn is an ACCEL device and cannot be disabled at runtime, hence two builds.
- Model: `/home/mkumar/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf` (bartowski, sha256 9da71c45...b6283).
- HF token in `~/.cache/huggingface/token` (exported as `HF_TOKEN` from `~/.bashrc`); never print it.

## Key findings (Q8_0, 14 threads, see results/dev_llamacpp_accel/)

- ZenDNN speeds up prefill ~2.5x (TTFT); decode (TPOT) is unchanged; throughput gain 1.19x (128-token prompts)
  to 1.78x (1024) and ~2x for batch. ZenDNN costs ~8 GB extra RSS (about one extra weight copy).
- Reason: `ggml_backend_zendnn_device_supports_op` adaptive fallback rejects MUL_MAT with `K <= 256`,
  `N <= 128` or `M <= 96` (N = tokens in the ubatch). Decode (N = active sequences) always falls back to
  ggml-cpu. "128" prompts are 129 tokens with BOS, so they just pass. `GGML_ZENDNN_ADAPTIVE_FALLBACK=0`
  disables the fallback (not yet benchmarked).
- Batch points with few requests (2 waves) trip the halves-based drift check by construction; TPOT drift is
  already non-gating for batch, tok/s drift is still gating.

## Status and next steps

See the progress log in the agent memory store; summary:

1. Done: full harness, restructure into packages, ZenDNN on/off support, HTML + CSV reports, Q8 comparison run.
2. Batch drift: switch to total-token drift and skip when a rep has fewer than ~4 waves.
3. Run a third config with `GGML_ZENDNN_ADAPTIVE_FALLBACK=0`.
4. BF16 comparison: needs 16 GB BF16 GGUF in `/tmp` (awaiting user OK); peak ~35 GB RAM with ZenDNN.
5. Fix the cut-off title in `plots.plot_acceleration`; rerun `pytest` after recent server/metrics edits.
6. vLLM + zentorch setup is on hold: the AMD-Zenai `vllm-zentorch` SKILL.md is behind SAML SSO, and disk is short.
