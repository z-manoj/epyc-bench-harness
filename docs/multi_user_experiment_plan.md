# Experiment plan: best multi-user (interactive) serving configuration

Goal: find the configuration that serves the **most concurrent chat users** of Llama 3.1 8B on this host while
every user still gets an acceptable experience, for vLLM and llama.cpp, with and without ZenDNN.

This differs from the [throughput plan](throughput_experiment_plan.md): peak tokens per second doesn't matter if
users wait 20 s for the first token. The metric here is **capacity at a latency target**: how many users, or
requests per second, the deployment sustains while meeting it. Instance splitting is shared with the throughput
plan; this plan adds latency targets, realistic traffic, and scheduler tuning.

**Target workload: prompts of about 8K tokens** (8192). This dominates every decision below.

## 0. What 8K prompts mean on this host

Measured, one request at a time, vLLM + zentorch on 96 cores (`results/sweep_vllm/`):

| Prompt | Precision | TTFT | Prompt processing rate | TPOT |
|---|---|---|---|---|
| 1024 | W8A8 | 0.61 s | ~1,680 tok/s | 35 ms |
| 4096 | W8A8 | 3.3 s | ~1,240 tok/s | 38 ms |
| 8192 | W8A8 | 9.2 s | ~890 tok/s | 40 ms |
| 1024 | BF16 | 0.99 s | ~1,030 tok/s | 58 ms |
| 4096 | BF16 | 4.8 s | ~850 tok/s | 60 ms |
| 8192 | BF16 | 11.9 s | ~690 tok/s | 62 ms |

Consequences:
- **TTFT floor is ~9 s (W8A8) or ~12 s (BF16)** for an uncached 8K prompt, with the whole socket working on that
  one request. No load level or tuning can go below this without making prefill itself faster.
- **Prompt processing gets slower per token as prompts grow** (attention cost grows with length): 8K runs at about
  half the per-token rate of 1K.
- **Prompt processing is the capacity ceiling.** Each uncached 8K prompt uses ~9 s of a full socket (W8A8). A socket
  can therefore start at most **~6-7 new uncached 8K requests per minute**, about 13 per minute for the host,
  before any time is spent generating. With users sending a request every ~1-2 minutes, that is roughly
  10-15 active users per socket.
- **Decode is not the problem.** TPOT stays at ~40 ms (W8A8) even with an 8K context, well inside targets.
- **KV cache**: ~1 GB per 8K-token sequence (BF16 KV), so 32 concurrent users need ~32-40 GB. Memory is plentiful,
  but set `VLLM_CPU_KVCACHE_SPACE` accordingly.
- **Prefix caching is the biggest lever.** If most of the 8K is shared (system prompt, RAG documents reused across
  questions, conversation history in multi-turn chat), only the new part is processed. A 90% cache hit turns a
  9 s prefill into ~1 s. How much of the real 8K is reusable must be established before anything else.
- **ZenDNN matters more here** than in the 1K results, because the workload is dominated by prompt processing,
  which is exactly what ZenDNN accelerates (llama.cpp ZenDNN halved TTFT at 1K). zentorch vs stock vLLM at 8K is
  unmeasured: stock vLLM crashed in the sweep.
- **Splitting a socket into instances hurts TTFT**: an 8K prompt on a 48-core instance takes roughly twice as long.
  Long prompts favour fewer, larger instances; tensor parallel across both sockets may even help TTFT.

## 1. Define the service targets first

Everything is judged against these, so fix them before measuring. Adjust to the product.

A fixed chat-style TTFT target (1-2 s) is impossible for uncached 8K prompts on this CPU. Targets are therefore set
relative to the number of tokens that actually need processing (the uncached part of the prompt), anchored to
the measured single-request floor:

| Metric | Interactive tier (p99) | Relaxed tier (p99) | What the user sees |
|---|---|---|---|
| TTFT | ≤ 2 s + 1.5 ms per uncached prompt token (≈ 14 s for a fully uncached 8K, ≈ 3 s for 1K uncached) | ≤ 30 s + 3 ms per uncached token | Delay before the answer starts |
| Time per output token (TPOT) | ≤ 100 ms (≥ 10 tok/s per user) | ≤ 200 ms | Streaming speed; people read ~5-8 tok/s |
| Inter-token gap | ≤ 500 ms | ≤ 2 s | Visible stalls mid-answer (a new 8K prompt entering the batch causes these) |
| Errors / timeouts | < 0.1% | < 0.1% | Failed requests |

A run **meets the targets** only if all rows pass. **Capacity** = the highest load that meets the targets. Report
both tiers, since the ranking can change. Also report the unloaded single-request TTFT per configuration as the
floor. If the product needs TTFT of a few seconds on 8K prompts, that is only reachable with high prefix-cache
hit rates, and the plan should confirm that early (stage 1).

## 2. Realistic traffic

Fixed-length prompts hide scheduler behaviour. Use traffic shaped like the real application:

- **Length mix**: prompts centred on 8K (e.g. 6-8K, a tail to 16K if the application allows it); outputs
  drawn from the expected answer lengths (e.g. 128-512 tokens). Keep a fixed 8192/256 workload for clean
  comparisons, and the old 1024/128 for continuity.
- **Cache-hit scenarios**, run as separate traffic sets because they change the answer completely:
  - 0% shared (every 8K prompt unique): worst case, prompt-processing bound.
  - ~50% shared (e.g. a 4K system prompt or document set shared by all users).
  - ~90% shared (multi-turn: each turn resends the history and adds ~500-800 new tokens).
  Measure the real application's hit rate if possible and weight the results by it.
- Prompt sets: `prompts/` only goes to 4096 tokens today; 8K sets with controllable shared prefixes need to be
  generated (the smoke client's token-ID prompts can already produce exact 8192-token requests).
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

- First, single-request floor at 8192 tokens for every candidate (TTFT, TPOT), including stock vLLM with
  `FREEZING=0` and llama.cpp. This alone may eliminate candidates and shows ZenDNN's prefill gain at 8K.
- One instance on socket 0 (96 cores), default scheduler settings, prefix caching on.
- Closed loop at 1, 2, 4, 8, 16, 32 users with think time; 8K traffic at 0%, 50% and 90% cache hit.
- Candidates: vLLM zentorch W8A8, vLLM stock W8A8 (`FREEZING=0`), vLLM zentorch BF16, llama.cpp ZenDNN Q8_0
  (`-np 32`), llama.cpp plain Q8_0.

Output: users supported at the target for each. Drop anything clearly behind (more than ~20% below the best).
Expected from earlier data: vLLM W8A8 leads; llama.cpp holds up at low user counts but its batching limits it.

### Stage 2: scheduler tuning (per surviving engine, one instance per socket)

Question: which scheduler settings maximise capacity?

vLLM, varied one at a time around the defaults, then a small grid over the two most sensitive:
- `--max-num-seqs`: 16, 32, 64, 128. Too low queues users (TTFT rises); too high slows every user's decode
  (TPOT rises).
- `--max-num-batched-tokens`: 1024, 2048, 4096, 8192, 16384. Controls how much prefill is mixed into each step.
  With 8K prompts this is the key TTFT vs TPOT trade-off: small chunks keep other users streaming smoothly but
  stretch each 8K prefill over more steps; large chunks finish prefills sooner but stall everyone's decode.
- Chunked prefill on vs off: with 8K prompts, off means one prompt blocks all decoding for ~9 s. Expected to be
  essential; verify with the inter-token gap metric.
- `--max-model-len` just above the longest prompt + output (e.g. 10-17K) and KV cache sized for users × length.
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
out, but each prefill runs on fewer cores, so TTFT for long prompts is worse. With 8K prompts, a 2 × 48 split
roughly doubles the TTFT floor (~18 s W8A8), so it likely only wins in the relaxed tier or at high cache-hit rates.
Round-robin also ignores load; test least-outstanding-requests routing too, and cache-aware (session-sticky)
routing for the multi-turn scenario, since a conversation only hits the prefix cache on the instance that holds it.

Also test **one instance across both sockets with tensor parallel 2** (`--tensor-parallel-size 2`,
`VLLM_CPU_OMP_THREADS_BIND="0-95|96-191"`): it could cut the 8K TTFT floor, at the cost of cross-socket
communication and host capacity. Compare against two independent per-socket instances.

### Stage 4: capacity search on the finalists

- Top 2-3 configurations from stages 2-3.
- Binary search on the Poisson rate (and on user count for closed loop) until the target is just met; confirm the
  final point with a 15-20 minute run.
- Repeat for the relaxed tier.

Output: the headline number per configuration, e.g. "vLLM zentorch W8A8, 2 × 48: N req/s or M users within
the interactive tier".

### Stage 5: robustness

On the chosen configuration:
- **Burst**: 3× rate for 30 s; time to recover to the target.
- **Longest prompts**: 10% of requests at 16K tokens (if the application allows it); effect on everyone else's
  TPOT and inter-token gap.
- **Cache pressure**: more distinct conversations than the prefix cache holds; watch the hit rate and TTFT fall.
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
