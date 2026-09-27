# CPU LLM Serving Benchmark Harness (llama.cpp + vLLM on ZenDNN)

Benchmarks Llama 3.1 8B serving on AMD EPYC: llama.cpp (`llama-server`, ggml-zendnn) and vLLM (zentorch),
with and without ZenDNN. Two parts:

- `scripts/`: setup for both engines and standalone smoke / sweep / offline batch / profiling tests. All current
  results on this host come from these.
- `bench/`: full harness (`bench` CLI), scenarios `online_single` / `online_multi` / `batch`, stability gating,
  resume, HTML report and CSVs. Its venv does not work on this host (see Environment).

Repo: `git@github-z-manoj:z-manoj/epyc-bench-harness.git` (branch `main`; the `github-z-manoj` SSH alias uses
`~/.ssh/id_ed25519_z-manoj`). User docs: `README.md`, `docs/deployment_recommendations.md`.

## Commands

```bash
# Setup (idempotent, no sudo)
scripts/setup_llamacpp.sh [--no-models] [--bf16] [--serial]      # build/ (ZenDNN) + build-nozendnn/
scripts/setup_vllm.sh [--only zentorch|cpu] [--no-models] [--rebuild-zentorch]
source ~/vllm-zen/env-zentorch.sh    # or env-cpu.sh; sets PATH, LD_PRELOAD, VLLM_CPU_*, FREEZING, AOT

# Tests (each writes results/<name>/)
scripts/smoke_1024_128.sh [config...]        # single stream 1024/128, 8 configs; smoke_report.py
scripts/sweep_vllm.sh [config...]            # SCENARIOS="prompt:gen:conc ..."; sweep_report.py
CPUS=0-95 MEM_NODE=0 N=50 FREEZING=0 CHUNKED=0 OUT=results/x scripts/run_offline_batch.sh cpu zentorch
scripts/amd_repro_throughput.sh [cpu-single cpu-multi zentorch-single zentorch-multi]
scripts/profile_vllm_server.sh <zentorch|cpu> <model_dir> <out_dir>

# Report scripts run in the tools venv
~/vllm-zen/tools/bin/python scripts/smoke_report.py --experiment <results_dir>   # same for sweep_report.py

# Harness (needs glibc >= 2.38; not usable on this host)
bench plan|run|resume|analyze|report|selftest ...   # see README
pytest -q                            # 26 tests; e2e tests are marked `slow`
```

## Layout

| Path | Role |
|---|---|
| `scripts/` | `setup_llamacpp.sh`, `setup_vllm.sh`, `smoke_1024_128.sh` + `smoke_client.py` + `smoke_report.py`, `sweep_vllm.sh` + `sweep_report.py`, `run_offline_batch.sh` + `offline_batch.py`, `amd_repro_throughput.sh`, `profile_vllm_server.sh`, `profile_vllm_ops.py` |
| `bench/core/` | `config.py` (pydantic models, `extra="forbid"`, matrix expansion, safe `eval_expr`), `status.py` (all status/flag constants), `io.py` (`log`, atomic `write_json`, `EventLog`) |
| `bench/system/` | `sampler.py` (separate process, 50 ms deadline-scheduled `/proc` sampler, Arrow IPC frames), `samples.py` (`SampleStore`, `finalize_samples` → `samples.parquet`), `env.py` (tunables, versions, checksums) |
| `bench/runner/` | `experiment.py`, `lifecycle.py` (`ServerConfigRunner`), `server.py` (`ServerGroup`: pinning, readiness, backend detection), `loadgen.py` (aiohttp streaming SSE), `dataset.py`, `storage.py` |
| `bench/analysis/` | `metrics.py`, `gates.py`, `consistency.py` (`decide_point`), `aggregate.py`, `acceleration.py` (ZenDNN vs baseline pairing), `recommend.py` |
| `bench/reporting/` | `doc.py` (Markdown + self-contained HTML), `plots.py`, `report.py` |
| `tests/` | `mock_server.py` (OpenAI-compatible mock), `test_config.py`, `test_analysis.py`, `test_e2e.py` |
| `examples/` | Harness configs; `experiment.yaml` documents every option. Core counts in them are for the old 8-core VM |
| `prompts/` | `prompts_{128..4096}.txt`, JSONL `{id, text}`, 20 prompts each |
| `docs/` | `deployment_recommendations.md` |
| `results/` | Committed: reports, CSVs, PNGs, raw JSONL. Ignored: `logs/`, `*.log`, parquet |

## Conventions

- Harness: every threshold lives in `core/config.py` `Thresholds` and is overridable from YAML. `decide_point` must
  stay a pure function of rep history (resume determinism). Status names come from `core/status.py`.
- Harness pairing: configs pair for speedup analysis when engine, precision, model, cores, args and env match,
  ignoring `binary` and env vars prefixed `ZENDNN`/`ZENTORCH`/`GGML_ZENDNN`, or via a shared `accel_group`.
- Backend verification: llama.cpp uses `llama-server --list-devices` (ZenDNN build lists a ZenDNN device, plain
  build lists none). vLLM: zentorch env must activate `ZenCpuPlatform`, stock env must stay on `CpuPlatform`.
- Scripts give each server its runtime env explicitly (vLLM via its env file, llama.cpp via `LLAMA_PRELOAD`) and
  `unset` the caller's `LD_PRELOAD` / `VLLM_*` first. Both engines run with the same libiomp5 + tcmalloc preload.
- Benchmarks send `cache_prompt: false` to llama.cpp and run vLLM with `--no-enable-prefix-caching`.
- Never `pkill -f` with a pattern that can match the calling shell's own command line; kill by PID.
- Never source an env file in the agent's own shell (it leaks `LD_PRELOAD`); use `bash -c "source ...; cmd"`.
- Use absolute `OUT` paths: scripts `cd /tmp` before launching vLLM.
- Comments state constraints only; match surrounding style.

## Environment (this host)

- 2× AMD EPYC 9R14 (Genoa, Zen 4), 96 cores per socket, SMT on (384 threads). NUMA0 = cpus 0-95 + 192-287,
  NUMA1 = cpus 96-191 + 288-383. CCD = 8 cores (CCD0 = 0-7 + 192-199). 1.5 TB RAM.
- Ubuntu 22.04, glibc 2.35, system gcc 11.4, no sudo. `numactl` available; cpufreq governor `performance`.
  ~266 GB free on `/`.
- Pinning: servers on socket 0 (`--physcpubind=0-95 --membind=0`), clients on 96-99. Idle root-owned Qwen vLLM
  containers (k3s) sit on cores 96-175; avoid socket 1 for timing-sensitive runs or check they are still idle.
- llama.cpp: `~/llama.cpp` (commit 9adc7f4). `build/bin/llama-server` (`GGML_ZENDNN=ON`, ZenDNN 6.1 bundled) and
  `build-nozendnn/bin/llama-server`. ggml-zendnn cannot be disabled at runtime, hence two builds.
- vLLM: micromamba envs `~/vllm-zen/env` (vLLM 0.28.0+cpu + zentorch 2.13.0.1, gcc/gxx 12) and
  `~/vllm-zen/env-cpu` (stock), torch 2.13.0+cpu. zentorch built from public `amd/ZenDNN-pytorch-plugin` tag
  `zentorch-2026-WW38` (source in `~/vllm-zen/zentorch-src`). vLLM 0.30 needs glibc 2.39.
- `~/vllm-zen/tools`: uv venv with aiohttp, pandas, matplotlib, numpy, tabulate (clients and reports).
  The repo `.venv` is broken here (its wheels need glibc 2.38).
- Models in `~/mkumar/models`: `Llama-3.1-8B-Instruct` (BF16 safetensors),
  `Meta-Llama-3.1-8B-Instruct-quantized.w8a8` (RedHatAI), `gguf/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf`,
  `gguf/Llama-3.1-8B-Instruct-BF16.gguf`.
- GitHub: `gh` CLI is logged in as `akhetan_amdeng`; SSH `github.com` = `sushant-zettabolt`; SSH alias
  `github-z-manoj` = `z-manoj`. None can access the private `AMD-Zenai` repos.
- HF token in `~/.cache/huggingface/token` (exported as `HF_TOKEN` from `~/.bashrc`); never print it. Tokens can
  appear in other users' `ps` output; never repeat them.

## Key findings

This host, socket 0, 1024-token prompt / 128 generated tokens (`results/smoke_1024_128/`):

- llama.cpp ZenDNN speeds up prefill ~2x (TTFT 1.97x BF16, 1.80x Q8_0); decode unchanged.
- vLLM zentorch vs stock: identical on a single stream (stock already uses oneDNN `onednn_mm`); prefill runs
  near peak (~17 TFLOPS), decode is memory-bandwidth bound.
- Offline batch (`results/offline_batch*/`): zentorch 1.06-1.07x over stock at 20 prompts, 1.035x at 50 prompts.
  Batch size 20 → 50 gives both ~24% more throughput. Freezing is worth ~1-2% on zentorch; chunked prefill ~0.
- Sweep (`results/sweep_vllm/`, zentorch only): W8A8 ~1.6x BF16 at 4/16/64 concurrent requests.
- AMD's published ~1.7x is two zentorch instances vs one stock instance on Turin (Zen 5); their like-for-like
  figures are 1.15-1.26x. The ZenDNN 5.1 blog compares against vLLM 0.9.0 + IPEX.
- llama.cpp fallback: `ggml_backend_zendnn_device_supports_op` rejects MUL_MAT with `K <= 256`, `N <= 128` or
  `M <= 96` (N = tokens in the ubatch), so decode always runs on ggml-cpu. `GGML_ZENDNN_ADAPTIVE_FALLBACK=0`
  disables the fallback (not yet benchmarked). ZenDNN costs ~8 GB extra RSS (one extra weight copy).

## Known issues

- Stock vLLM 0.28.0 intermittently segfaults in `onednn_mm` (`dnnl_helper.cpp:146`, `get_runtime_memory_ptr`)
  during compile warmup with `TORCHINDUCTOR_FREEZING=1`. Disabling chunked prefill does not help. Stack matches
  vllm-project/vllm#46131 (open, fix PR #46154 unmerged), but our crash happens before any mixed batch.
  Workaround: `FREEZING=0` (env files honour a pre-set `TORCHINDUCTOR_FREEZING`). zentorch never crashed.
- `scripts/amd_repro_throughput.sh` uses a relative default `OUT` while instances run from `/tmp`; JSON writes fail.
- `vllm` per-request `seed` fails on CPU ("CPU Generator does not use offset"); `offline_batch.py` omits it.

## Status and next steps

1. Done: harness, setup scripts, llama.cpp + vLLM smoke (8 configs), vLLM sweep (zentorch), offline batch
   20 and 50 prompts, README, deployment recommendations, repo pushed.
2. W8A8 offline batch, zentorch vs stock (`FREEZING=0`).
3. Two vLLM instances per socket (AMD methodology); fix `OUT` in `amd_repro_throughput.sh` first.
4. llama.cpp with `GGML_ZENDNN_ADAPTIVE_FALLBACK=0`, single stream and 16 parallel slots.
5. llama.cpp parallel slots (`-np`) under concurrent load; stock vLLM sweep with `FREEZING=0`.
6. Harness: port examples to this host's core layout; batch drift check (total-token drift, skip under ~4 waves);
   cut-off title in `plots.plot_acceleration`.
7. Private `AMD-Zenai/ZenDNN_PyTorch_Plugin`: needs an org owner to grant one of the accounts read access.
