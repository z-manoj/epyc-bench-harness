# epyc-bench-harness

Benchmarks for serving Llama 3.1 8B on AMD EPYC CPUs with **llama.cpp** and **vLLM**, each with and without
AMD's ZenDNN acceleration (ggml-zendnn for llama.cpp, the zentorch plugin for vLLM).

The repo has two parts:

- **Standalone scripts** (`scripts/`): setup for both engines, plus smoke, sweep, offline batch and profiling
  tests. All results below come from these.
- **A full benchmark harness** (`bench/`): a Python package with a `bench` CLI that runs a configurable experiment
  matrix (single-stream, multi-stream and batch scenarios) with stability gating, repetitions, resume support
  and HTML/CSV reports.

## Test host

| | |
|---|---|
| CPU | 2× AMD EPYC 9R14 (Genoa, Zen 4), 96 cores per socket, SMT on (384 threads) |
| NUMA | node 0 = cpus 0-95, 192-287; node 1 = cpus 96-191, 288-383 |
| Memory | 1.5 TB |
| OS | Ubuntu 22.04 (glibc 2.35), no sudo |

All tests pin the server to socket 0 (cores 0-95, memory on node 0) with `numactl`.

## Software versions

| Component | Version |
|---|---|
| llama.cpp | commit `9adc7f4`, built twice: `build/` (`GGML_ZENDNN=ON`, ZenDNN 6.1 bundled) and `build-nozendnn/` |
| vLLM | 0.28.0+cpu wheel (0.30 needs glibc 2.39) |
| PyTorch | 2.13.0+cpu |
| zentorch | 2.13.0.1, built from public [amd/ZenDNN-pytorch-plugin](https://github.com/amd/ZenDNN-pytorch-plugin) tag `zentorch-2026-WW38` |
| Runtime | `LD_PRELOAD` of llvm-openmp `libiomp5` + `libtcmalloc_minimal` for both engines |

Models: `Meta-Llama-3.1-8B-Instruct` (BF16 safetensors and GGUF), `Q8_0` GGUF, and RedHatAI `w8a8` INT8.

## Setup

Both scripts are idempotent and need no root.

```bash
# llama.cpp: clone, build with and without ZenDNN in parallel, verify backends, download GGUF models
scripts/setup_llamacpp.sh [--no-models] [--bf16] [--serial]

# vLLM: two micromamba envs in ~/vllm-zen (env = vLLM + zentorch, env-cpu = stock vLLM),
# plus runtime env files env-zentorch.sh and env-cpu.sh
scripts/setup_vllm.sh [--only zentorch|cpu] [--no-models] [--rebuild-zentorch]
```

Key overrides: `LLAMA_DIR`, `LLAMA_COMMIT`, `MODELS_DIR`, `VLLM_BASE`, `VLLM_VERSION`, `ZENTORCH_REPO`,
`ZENTORCH_REF`. Gated Hugging Face models need `HF_TOKEN` set (or `~/.cache/huggingface/token`).

To run anything in a vLLM env:

```bash
source ~/vllm-zen/env-zentorch.sh   # or env-cpu.sh for stock vLLM
```

The env files set `LD_PRELOAD`, `VLLM_CPU_KVCACHE_SPACE` (default 90 GB), `VLLM_CPU_OMP_THREADS_BIND`
(default 0-95), `TORCHINDUCTOR_FREEZING` (default 1) and `VLLM_USE_AOT_COMPILE` (default 0). All can be
overridden by exporting them first.

## Running the tests

Each script writes into `results/<name>/` and prints a summary.

| Script | What it measures |
|---|---|
| `scripts/smoke_1024_128.sh [config...]` | Single stream, 1024-token prompt, 128 generated tokens, 3 runs, all 8 engine/precision configs. `scripts/smoke_report.py` builds the report. |
| `scripts/sweep_vllm.sh [config...]` | vLLM with vs without zentorch on long prompts (4096, 8192) and 4/16/64 concurrent requests. Scenarios set via `SCENARIOS="prompt:gen:concurrency ..."`. `scripts/sweep_report.py` builds the report. |
| `scripts/run_offline_batch.sh [zentorch cpu]` | Offline `LLM.generate` over `N` real-text prompts of `PROMPT_LEN` tokens. Options: `CPUS`, `MEM_NODE`, `N`, `RUNS`, `FREEZING`, `CHUNKED`. |
| `scripts/amd_repro_throughput.sh` | AMD blog methodology: `vllm bench throughput`, 128 in / 128 out, 1 instance vs 2 instances. |
| `scripts/profile_vllm_server.sh`, `scripts/profile_vllm_ops.py` | Torch profiler op breakdown for one request. |

Example, the recommended offline batch comparison:

```bash
CPUS=0-95 MEM_NODE=0 N=50 FREEZING=0 CHUNKED=0 OUT=results/offline_batch_n50 \
  scripts/run_offline_batch.sh cpu zentorch
```

## Results summary

### Single stream, 1024-token prompt, 128 generated tokens

| Config | Time to first token | Time per output token |
|---|---|---|
| llama.cpp BF16, ZenDNN | 1.29 s | 50.5 ms |
| llama.cpp BF16, plain CPU | 2.55 s | 51.1 ms |
| vLLM BF16, zentorch | 0.99 s | 57.6 ms |
| vLLM BF16, stock | 1.02 s | 56.3 ms |
| llama.cpp Q8_0, ZenDNN | 1.26 s | 31.0 ms |
| llama.cpp Q8_0, plain CPU | 2.27 s | 30.9 ms |
| vLLM W8A8, zentorch | 0.61 s | 35.2 ms |
| vLLM W8A8, stock | 0.64 s | 35.0 ms |

- ZenDNN speeds up llama.cpp prefill about 2x (1.97x BF16, 1.80x Q8_0). Decode is unchanged because it is
  memory-bandwidth bound and ggml-zendnn falls back to ggml-cpu kernels for small matmuls.
- vLLM is the same with and without zentorch on a single stream: stock vLLM already uses oneDNN matmuls.

### Offline batch, BF16, 1024-token prompts, 128 generated tokens (total tokens/s, median of 3)

| Batch | Freezing | Chunked prefill | zentorch | stock vLLM | Speedup |
|---|---|---|---|---|---|
| 20 | on | on | 613 | crashes | n/a |
| 20 | off | on | 605 | 570 | 1.06x |
| 20 | off | off | 594 | 557 | 1.07x |
| 50 | off | off | 729 | 704 | 1.035x |

### Why this is far from AMD's published ~1.7x

- AMD's headline 1.68x ([ZenDNN 5.2 blog](https://www.amd.com/en/developer/resources/technical-articles/2026/zendnn-5-2-accelerating-vllm-inference-on-amd-epyc-cpus.html))
  compares **two** zentorch instances with **one** stock instance. Like-for-like gains in AMD's own footnotes
  are 1.15x (one instance) and 1.26x (two instances).
- The [ZenDNN 5.1 blog](https://www.amd.com/en/developer/resources/technical-articles/2025/zendnn-5-1-brings-new-optimizations.html)
  (1.10-1.53x) compares against vLLM 0.9.0 + IPEX, a much weaker baseline than today's stock vLLM CPU backend.
- AMD tested on EPYC 9755 (Turin, Zen 5), whose full-width AVX-512 suits ZenDNN kernels better than Zen 4.

## Known issues

- **Stock vLLM segfaults with `TORCHINDUCTOR_FREEZING=1`.** The crash is in `onednn_mm`
  (`dnnl_helper.cpp:146`, `get_runtime_memory_ptr`) during compile warmup, and is intermittent. Turning
  chunked prefill off does not help. The stack matches
  [vllm-project/vllm#46131](https://github.com/vllm-project/vllm/issues/46131), but our crash happens before any
  mixed batch runs. Workaround: `FREEZING=0`, which costs zentorch only about 1-2%.
- `scripts/amd_repro_throughput.sh` writes its JSON output relative to `/tmp`; pass an absolute `OUT`.
- The private `AMD-Zenai/ZenDNN_PyTorch_Plugin` repo is not accessible, so zentorch is built from the public repo.

## Full harness (`bench` CLI)

Requires Python 3.12+ and glibc 2.38+ for the pinned wheels.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[test]'

bench plan --config examples/local_llamacpp_q8.yaml [-v]   # expanded matrix and wall-time estimate
bench run --config <config.yaml> [--only <server_config_id>...]
bench resume --experiment results/<experiment_id>          # continues without duplicating reps
bench analyze --experiment <dir>
bench report --experiment <dir> [--no-html] [--timelines all|flagged|none]
bench selftest                                             # sampler jitter + mock end-to-end run
pytest -q                                                  # e2e tests are marked slow
```

Configs live in `examples/`; `examples/experiment.yaml` documents every option. Thresholds are all overridable
from YAML. See `CLAUDE.md` for the package layout and conventions.

## Repository layout

| Path | Contents |
|---|---|
| `bench/` | Harness package: `core/` config and status, `system/` sampler and env capture, `runner/` server lifecycle and load generator, `analysis/` metrics and gating, `reporting/` HTML/Markdown/plots |
| `scripts/` | Setup, smoke, sweep, offline batch, AMD reproduction and profiling scripts |
| `tests/` | Unit and end-to-end tests with an OpenAI-compatible mock server |
| `examples/` | Harness experiment configs |
| `prompts/` | Prompt sets of 128-4096 tokens (JSONL, 20 prompts each) |
| `results/` | Reports, CSVs, charts and raw per-request JSONL for each test (server logs are not committed) |
