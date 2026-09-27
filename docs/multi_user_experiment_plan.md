# Experiment plan: best multi-user (interactive) serving configuration

Goal: find the configuration that serves the **most concurrent chat users** of Llama 3.1 8B on this host while
every user still gets an acceptable experience, for vLLM and llama.cpp, with and without ZenDNN.

This differs from the [throughput plan](throughput_experiment_plan.md): peak tokens per second doesn't matter if
users wait 20 s for the first token. The metric here is **capacity at a latency target**: how many users, or
requests per second, the deployment sustains while meeting it. Instance splitting is shared with the throughput
plan; this plan adds latency targets, realistic traffic, and scheduler tuning.

## 1. Define the service targets first

Everything is judged against these, so fix them before measuring. Suggested defaults for chat; adjust to the
product:

| Metric | Target (p99 unless stated) | What the user sees |
|---|---|---|
| Time to first token (TTFT) | ≤ 2 s (p50 ≤ 1 s) | Delay before the answer starts |
| Time per output token (TPOT) | ≤ 100 ms (≥ 10 tok/s per user) | Streaming speed; people read ~5-8 tok/s |
| Inter-token gap | p99 ≤ 300 ms | Visible stalls mid-answer |
| Errors / timeouts | < 0.1% | Failed requests |

A run **meets the targets** only if all rows pass. **Capacity** = the highest load that meets the targets.
Report capacity for a relaxed tier too (e.g. TTFT ≤ 5 s, TPOT ≤ 200 ms), since the ranking can change.

## 2. Realistic traffic

Fixed-length prompts hide scheduler behaviour. Use traffic that looks like chat:

- **Length mix**: prompt and output lengths drawn from a distribution, e.g. ShareGPT-style
  (median prompt ~300-500 tokens with a long tail to 4K; median output ~200-300 tokens). Also keep the fixed
  1024/128 workload for comparison with earlier results.
- **Shared prefixes**: most chat requests share a system prompt (e.g. 500-1500 tokens), and multi-turn
  conversations resend the history. Model both, because prefix caching changes the result a lot.
- **Arrival model**, two kinds:
  - **Closed loop** (N users, each sends, waits for the full answer, "thinks" for a few seconds, repeats). Maps
    directly to "how many users can we host". The harness supports closed loop today (without think time).
  - **Open loop** (Poisson arrivals at a fixed request rate, independent of how fast the server is). Shows queue
    build-up and tail latency honestly; needed for capacity in requests per second.
- **Bursts**: a short spike (e.g. 3× the base rate for 30 s) to check recovery.

## 3. Configuration space

| Dimension | Options |
|---|---|
| Engine | vLLM, llama.cpp |
| Acceleration | zentorch / stock (vLLM); ZenDNN / plain build (llama.cpp) |
| Precision | W8A8 or Q8_0 (INT8), BF16 |
| Instances per socket | 1 × 96, 2 × 48, 4 × 24, 6 × 16 cores (CCD-aligned) |
| vLLM scheduler | `--max-num-seqs` 8-128; `--max-num-batched-tokens` 1024-8192; chunked prefill on/off; prefix caching on/off; `TORCHINDUCTOR_FREEZING` |
| vLLM memory | `VLLM_CPU_KVCACHE_SPACE` sized to users × context |
| llama.cpp | `-np` (parallel slots) 4-32; `-c` = slots × per-user context; `-b` / `-ub` (keep `-ub` > 128 for ZenDNN); prompt caching on; `GGML_ZENDNN_ADAPTIVE_FALLBACK` 1/0 |

The full product is far too large, so the stages below prune it step by step.

## 4. Stages

### Stage 0: harness readiness (tooling)

- Get the `bench` harness running on this host: rebuild `.venv` with Python 3.12 from wheels compatible with
  glibc 2.35 (the current venv was built for glibc 2.38). It already has multi-instance launch, round-robin
  endpoints, closed-loop load and target-based goodput (`SLA`, `add_goodput`).
- Add to the load generator: open-loop Poisson mode, think time for closed loop, length distributions, and
  shared-prefix / multi-turn prompt sets.
- Add metrics: inter-token gap p99, queue time (request arrival to first scheduled), and per-request TTFT split into
  queueing and prefill if the engine exposes it.

### Stage 1: engine and precision screen

Question: which engine/precision combinations are worth tuning?

- One instance on socket 0 (96 cores), default scheduler settings, prefix caching on.
- Closed loop at 1, 4, 8, 16, 32, 64 users; ShareGPT-style mix with a shared system prompt.
- Candidates: vLLM zentorch W8A8, vLLM stock W8A8 (`FREEZING=0`), vLLM zentorch BF16, llama.cpp ZenDNN Q8_0
  (`-np 32`), llama.cpp plain Q8_0.

Output: users supported at the target for each. Drop anything clearly behind (more than ~20% below the best).
Expected from earlier data: vLLM W8A8 leads; llama.cpp holds up at low user counts but its batching limits it.

### Stage 2: scheduler tuning (per surviving engine, one instance per socket)

Question: which scheduler settings maximise capacity?

vLLM, varied one at a time around the defaults, then a small grid over the two most sensitive:
- `--max-num-seqs`: 16, 32, 64, 128. Too low queues users (TTFT rises); too high slows every user's decode
  (TPOT rises).
- `--max-num-batched-tokens`: 1024, 2048, 4096, 8192. Controls how much prefill is mixed into each step; small
  values protect TPOT, large values reduce TTFT.
- Chunked prefill on vs off: expected to matter for TTFT/TPOT tails with long prompts even though it made no
  throughput difference.
- Prefix caching on vs off with the shared-prefix traffic.
- zentorch with freezing on vs off.

llama.cpp:
- `-np` 8, 16, 32 with `-c` scaled to match; `-ub` 256 / 512; `-b` 1024 / 2048.
- `GGML_ZENDNN_ADAPTIVE_FALLBACK=1` vs `0` (with many slots, decode matmuls get bigger and ZenDNN might help).

Output: tuned settings per engine, and which knob mattered most.

### Stage 3: instance split under multi-user load

Question: at the tuned settings, is one big instance or several smaller ones better for users?

- 1 × 96, 2 × 48, 4 × 24 (and 6 × 16 if 4 × 24 wins), requests round-robined. Rescale `--max-num-seqs` and KV
  cache per instance.
- Open-loop Poisson sweep of the request rate; find the capacity (requests/s at the target) for each split.

Trade-off to watch: several smaller instances each run smaller batches, so TPOT is better and queueing spreads
out, but each prefill runs on fewer cores, so TTFT for long prompts is worse. Round-robin also ignores load;
if splits look promising, test least-outstanding-requests routing too.

### Stage 4: capacity search on the finalists

- Top 2-3 configurations from stages 2-3.
- Binary search on the Poisson rate (and on user count for closed loop) until the target is just met; confirm the
  final point with a 15-20 minute run.
- Repeat for the relaxed tier.

Output: the headline number per configuration, e.g. "vLLM zentorch W8A8, 2 × 48: 2.4 req/s or ~60 users at
p99 TTFT ≤ 2 s".

### Stage 5: robustness

On the chosen configuration:
- **Burst**: 3× rate for 30 s; time to recover to the target.
- **Long-prompt mix**: 10% of requests at 4-8K tokens; effect on everyone else's TPOT.
- **Soak**: 2 hours at 80% of capacity; watch memory growth, latency drift, crashes (stock vLLM especially).
- **Both sockets**: replicate per socket; confirm about 2× capacity.

### Stage 6: ZenDNN value at the chosen configuration

Rerun the final configuration with zentorch vs stock vLLM and ZenDNN vs plain llama.cpp, same traffic and
settings. Report capacity gain at the target, not peak throughput gain: prefill acceleration mainly improves TTFT,
so ZenDNN's value for users may be larger than its throughput gain suggests.

## 5. Controls

- Same pinning rules as the throughput plan: whole-CCD blocks, memory on the local node, client on separate cores,
  k3s Qwen containers idle.
- Warm up each server with a few minutes of traffic before measuring (compile, caches, prefix cache fill).
- Minimum 5-minute measured window per point (tails need samples: p99 on under ~1000 requests is noisy), 2
  repetitions, and rerun if capacity differs by more than 5%.
- Fixed random seed for the traffic generator so every configuration sees the same request sequence.

## 6. Decision rules

- Pick the highest capacity at the interactive target. If two configurations are within 5%, prefer the simpler
  one (fewer instances, default settings).
- Check the relaxed-tier ranking; if it differs, document both.
- Reject any configuration that failed the soak or burst tests, whatever its capacity.

## 7. Time budget

| Stage | Time |
|---|---|
| 0 Tooling | 1-2 days of development |
| 1 Engine screen | 3-4 h |
| 2 Scheduler tuning | 6-8 h |
| 3 Instance split | 4-5 h |
| 4 Capacity search | 3-4 h |
| 5 Robustness | 4 h (includes 2 h soak) |
| 6 ZenDNN value | 2 h |

About 22-27 hours of machine time after the tooling is in place.
