# Deployment recommendations: Llama 3.1 8B on AMD EPYC (Genoa)

Suggested configurations for llama.cpp and vLLM, with and without ZenDNN, for three deployment shapes:
single user, multiple interactive users, and offline batch.

Every recommendation is marked as **measured** (backed by results in this repo) or **untested** (judgement,
not yet verified on this host).

Test host: 2× EPYC 9R14 (Zen 4, 96 cores per socket), 1.5 TB RAM. Servers pinned to socket 0 (cores 0-95,
memory on node 0). Versions: llama.cpp `9adc7f4`, vLLM 0.28.0+cpu, zentorch 2.13.0.1, torch 2.13.0+cpu.

## Summary

| Deployment | Engine | Weights | ZenDNN / zentorch | Key settings |
|---|---|---|---|---|
| Single user | vLLM or llama.cpp | INT8 (W8A8 or Q8_0) | Essential for llama.cpp prompt speed; no effect for vLLM | 96 threads on one socket, prompt caching on |
| Multiple users | vLLM | W8A8 | Small gain, low risk | 16-32 sequences, chunked prefill on, KV cache sized for users × context |
| Offline batch | vLLM | W8A8 or BF16 | 3-7% gain | Largest batch that fits, one process per socket |

- INT8 weights are the biggest single win: token generation is memory-bandwidth bound, and INT8 halves the bytes
  read per token (about 1.6x faster than BF16).
- ZenDNN only accelerates large matrix multiplies, which mostly happen during prompt processing. How much it
  helps depends on how much of the workload is prompt processing.
- Batch size matters more than the engine choice for throughput.

## What ZenDNN accelerates

| | llama.cpp + ggml-zendnn | vLLM + zentorch |
|---|---|---|
| Prompt processing | About 2x faster (time to first token 1.26 s vs 2.27 s, Q8_0) | Same as stock vLLM, which already uses oneDNN |
| Token generation | No change: small matmuls fall back to ggml-cpu | No change on a single stream |
| Batched throughput | Not measured on this host | 1.035-1.07x over stock |
| Cost | About 8 GB extra RAM (an extra copy of the weights) | Longer startup compile |

In llama.cpp, the ZenDNN matmul only runs when a chunk has more than 128 tokens and the weight dimensions are
large enough (K > 256, M > 96). Prompt chunks qualify; generating one token per user never does.

## Single user

One conversation at a time; latency matters most.

**Recommendation: INT8 weights on either engine, pinned to one socket.**

Measured, 1024-token prompt, 128 generated tokens:

| Engine and weights | Time to first token | Time per output token |
|---|---|---|
| vLLM W8A8, zentorch | 0.61 s | 35.2 ms |
| vLLM W8A8, stock | 0.64 s | 35.0 ms |
| llama.cpp Q8_0, ZenDNN | 1.26 s | 31.0 ms |
| llama.cpp Q8_0, plain CPU | 2.27 s | 30.9 ms |
| vLLM BF16, zentorch | 0.99 s | 57.6 ms |
| llama.cpp BF16, ZenDNN | 1.29 s | 50.5 ms |

- vLLM gives the fastest first token. llama.cpp generates slightly faster and is a single binary with no Python
  stack.
- If using llama.cpp, always use the ZenDNN build: it halves time to first token.

llama.cpp (ZenDNN build):

```bash
LD_PRELOAD=$HOME/vllm-zen/env/lib/libiomp5.so:$HOME/vllm-zen/env/lib/libtcmalloc_minimal.so.4 \
numactl --physcpubind=0-95 --membind=0 \
  ~/llama.cpp/build/bin/llama-server -m Meta-Llama-3.1-8B-Instruct-Q8_0.gguf -t 96 -c 8192
```

- Keep `-ub` (micro-batch) at the default 512. It must stay above 128 or ZenDNN is skipped entirely.
- Keep prompt caching on (the default). Benchmarks disable it; real chats reuse the conversation prefix.

vLLM (zentorch):

```bash
source ~/vllm-zen/env-zentorch.sh
export VLLM_CPU_OMP_THREADS_BIND=0-95 VLLM_CPU_KVCACHE_SPACE=8
numactl --membind=0 vllm serve <llama-3.1-8b-w8a8> --max-num-seqs 4 --max-model-len 8192
```

Untested: fewer cores (32-48) would probably generate just as fast, since generation is bandwidth-bound, and would
free the rest of the socket.

## Multiple users

Interactive serving with several conversations at once.

**Recommendation: vLLM with W8A8 and zentorch, one instance per socket.**

Measured, vLLM + zentorch, 1024-token prompts, output tokens per second:

| Concurrent users | W8A8 | BF16 |
|---|---|---|
| 4 | 66 | 42 |
| 16 | 126 | 78 |
| 64 | 163 | 101 |

```bash
source ~/vllm-zen/env-zentorch.sh
export VLLM_CPU_OMP_THREADS_BIND=0-95 VLLM_CPU_KVCACHE_SPACE=24
numactl --membind=0 vllm serve <llama-3.1-8b-w8a8> --max-num-seqs 32 --max-model-len 4096
```

- `--max-num-seqs` around 16-32. Past about 16 users, total throughput grows slowly while each user's generation
  speed drops.
- Keep chunked prefill on: it stops long prompts from stalling other users, and turning it off gained nothing.
- Keep prefix caching on.
- Size the KV cache: Llama 8B needs about 128 KB per token in BF16, so 32 users × 4K context is about 16 GB.
- zentorch: `TORCHINDUCTOR_FREEZING=1` is stable. Stock vLLM: set `TORCHINDUCTOR_FREEZING=0`, or it can crash on
  startup (see Known issues).
- Second socket: run a separate instance bound to cores 96-191 and memory node 1, behind a load balancer. Don't
  spread one instance across both sockets.

Untested:
- Stock vLLM under concurrent load (it crashed in the sweep), so zentorch's multi-user gain isn't measured.
  Going by the batch results it is probably around 5%.
- llama.cpp with many users: `-np <users> -c <users × context>`. Its prompt speedup still applies, but generation
  for all users stays on plain CPU kernels, and its batching is weaker than vLLM's.
- llama.cpp with `GGML_ZENDNN_ADAPTIVE_FALLBACK=0`, which forces ZenDNN for small matmuls. With many users the
  generation matmuls get larger (one row per active user), so ZenDNN might start helping generation.

## Offline batch

Throughput only, no latency target.

**Recommendation: vLLM `LLM.generate` with zentorch and the largest batch that fits.**

Measured, BF16, 1024-token prompts, 128 generated tokens, total tokens per second (median of 3 runs):

| Batch | Freezing | Chunked prefill | zentorch | stock vLLM | Speedup |
|---|---|---|---|---|---|
| 20 | on | on | 613 | crashes | n/a |
| 20 | off | on | 605 | 570 | 1.06x |
| 20 | off | off | 594 | 557 | 1.07x |
| 50 | off | off | 729 | 704 | 1.035x |

```bash
CPUS=0-95 MEM_NODE=0 N=50 FREEZING=1 scripts/run_offline_batch.sh zentorch
```

- Going from 20 to 50 prompts gave both engines about 24% more throughput. Keep increasing (64-128 or more) until
  the KV cache fills, at about 150 MB per 1K-token request.
- `max_num_seqs` of at least the batch size; `max_num_batched_tokens` 4096 or more.
- zentorch: `TORCHINDUCTOR_FREEZING=1` adds about 1-2%. Stock vLLM: keep it at 0.
- Chunked prefill makes no difference either way.
- Both sockets: one process per socket, split the work between them.

Untested:
- W8A8 in offline mode; the concurrency sweep suggests roughly 1.5x over BF16.
- AMD's two-instances-per-socket setup, which their footnotes credit with part of their gain.
- llama.cpp with ZenDNN gave about 2x batch throughput over its own plain CPU build on an earlier 8-core VM,
  because batches are mostly prompt processing. vLLM is still the better tool for pure throughput.

## ZenDNN and zentorch settings

| Setting | Engine | Recommended | Why |
|---|---|---|---|
| ZenDNN build (`build/`) | llama.cpp | Always | ZenDNN can't be switched on or off at runtime |
| `-ub` (micro-batch) | llama.cpp | Above 128 (default 512 is fine) | Below that, ZenDNN is skipped |
| `GGML_ZENDNN_ADAPTIVE_FALLBACK` | llama.cpp | Default (1); try 0 for many users | 0 forces ZenDNN for small matmuls |
| `TORCHINDUCTOR_FREEZING` | vLLM | 1 with zentorch, 0 with stock | Lets zentorch prepack weights; stock vLLM can crash with 1 |
| `VLLM_USE_AOT_COMPILE` | vLLM | 0 | Conflicts with freezing |
| `ZENDNNL_MATMUL_ALGO` | vLLM | Default | zentorch README recommendation |
| `ZENDNNL_MATMUL_WEIGHT_CACHE` | vLLM | Default (on) | Only disabled for zentorch unit tests |
| `LD_PRELOAD` libiomp5 + tcmalloc | Both | Always | Same runtime for both engines; set by the env files |
| `VLLM_CPU_OMP_THREADS_BIND` | vLLM | Physical cores of one socket (0-95) | One OpenMP thread per core |
| Pinning | Both | One socket's physical cores, memory on the same node | Crossing sockets costs memory bandwidth |

## Known issues

- Stock vLLM 0.28.0 segfaults in `onednn_mm` during compile warmup when `TORCHINDUCTOR_FREEZING=1`. It is
  intermittent and not fixed by disabling chunked prefill. The stack matches
  [vllm-project/vllm#46131](https://github.com/vllm-project/vllm/issues/46131). Workaround: `FREEZING=0`.
- zentorch's gains here (1.035-1.07x) are well below AMD's published figures. AMD's 1.68x compares two zentorch
  instances against one stock instance on Turin (Zen 5); their like-for-like figures are 1.15-1.26x. Their ZenDNN
  5.1 figures compare against vLLM 0.9.0 + IPEX, a weaker baseline than today's stock vLLM.

## Gaps to close before production

In order of how likely they are to change a recommendation:

1. W8A8 in offline batch mode.
2. Two vLLM instances per socket.
3. llama.cpp with `GGML_ZENDNN_ADAPTIVE_FALLBACK=0`, single user and 16 parallel users.
4. llama.cpp with parallel slots (`-np`) under concurrent load.
5. Core-count scaling for a single user.
6. Stock vLLM under concurrent load (with `FREEZING=0`).
