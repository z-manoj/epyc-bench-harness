# Experiment plan: best throughput deployment (single vs multi-instance)

Goal: find how to split a 2-socket EPYC host into serving instances to get the most total throughput for
Llama 3.1 8B, for vLLM and llama.cpp, with and without ZenDNN, and at what latency cost.

The question is not only "one instance or two": it is **how many instances, how many cores each, where they sit,
and how much load each gets**. All of these interact, so the plan varies them in stages instead of as one huge grid.

## Why multiple instances can win

- **Scaling limits inside one instance.** Past some core count, adding cores to one instance gives little: OpenMP
  barriers per layer, thread synchronisation, and Python/scheduler overhead in vLLM grow with thread count.
  Several smaller instances each run in their efficient range.
- **Cache locality.** Genoa has 12 CCDs of 8 cores per socket, each with its own 32 MB L3. An instance confined to
  a few CCDs keeps its working set in fewer L3 slices.
- **Decode is memory-bandwidth bound.** Each instance reads the full weights once per decode step, whatever its
  batch. More instances means more weight reads per second in total, so multi-instance helps only while
  memory bandwidth isn't saturated. This is the main counterweight.
- **Batching efficiency.** One large instance batches more sequences per weight read. Splitting the same load across
  instances shrinks each batch.

The best split is where these balance, and it differs by engine, precision, prompt length and load. It has to be
measured.

## Host facts that shape the design

| | |
|---|---|
| Sockets / NUMA | 2 sockets, 1 NUMA node each (NPS1) |
| Cores | 96 per socket = 12 CCDs × 8 cores; SMT siblings at +192 |
| L3 | 32 MB per CCD |
| Memory | 12 DDR5 channels per socket; 1.5 TB total |
| Model size | BF16 ~16 GB, INT8 ~8.5 GB per instance copy (plus ~8 GB for llama.cpp ZenDNN) |

Memory capacity is not a constraint: even 24 BF16 instances fit. Memory bandwidth is.

## Fixed controls for every run

- Pin with `numactl --physcpubind=<cores> --membind=<node>`; physical cores only unless the run is testing SMT.
- Instance core blocks are **whole CCDs** (multiples of 8 cores, starting on a CCD boundary) unless the run is
  testing placement.
- Client / load generator on cores outside every instance (e.g. the last CCD of socket 1, or SMT siblings of
  unused cores), and never on a socket under test if avoidable.
- vLLM: `TORCHINDUCTOR_FREEZING=0` for stock vLLM (it crashes with 1); test zentorch with 1. Prefix caching
  off for benchmarking. Same env file preload for both engines.
- llama.cpp: `cache_prompt: false`, ZenDNN build vs `build-nozendnn`.
- 1 warmup + 3 measured runs per point; report median and spread. Rerun a point if the coefficient of variation
  is above 3%.
- Interleave configurations (A, B, A, B) rather than running all of A then all of B, so drift affects both.
- Check the host is quiet first: the root-owned k3s Qwen containers on cores 96-175 must be idle (`top`).
- Record per point: total throughput (output and total tok/s), requests/s, TTFT and TPOT percentiles (p50, p99),
  CPU utilisation, and memory bandwidth if a counter is available.

## Workloads

Throughput depends heavily on the prompt/output mix, so every stage uses the same small set:

| Name | Prompt / output tokens | Character |
|---|---|---|
| `short` | 128 / 128 | Balanced, decode-heavy (AMD's blog workload) |
| `prefill` | 1024 / 128 | Prefill-heavy (our main workload so far) |
| `decode` | 128 / 512 | Decode-heavy, stresses memory bandwidth |

Load model:
- **Offline**: a fixed pool of requests (e.g. 512 per socket) split evenly across instances; measure wall time to
  finish all. This gives peak throughput.
- **Online**: Poisson arrivals at a target request rate, requests round-robined across instances; sweep the rate.
  This gives throughput at a latency target (goodput).

## Stage 1: single-instance core scaling

Question: how does one instance's throughput grow with cores, and where does it flatten?

- Cores: 8, 16, 24, 32, 48, 64, 96 (CCD-aligned blocks on socket 0).
- Offline load, batch large enough to saturate (`max_num_seqs` 128 for vLLM, `-np 32` for llama.cpp).
- Engines: vLLM zentorch, vLLM stock, llama.cpp ZenDNN, llama.cpp plain CPU. BF16 and INT8.
- Workloads: all three.

Output: throughput vs cores and throughput per core. The knee of the curve is the natural instance size candidate.
If throughput per core stays flat up to 96, a single instance per socket is likely best and later stages can be
cut down.

Cost: 7 core counts × 4 engines × 2 precisions × 3 workloads ≈ 168 points; at 5-10 min each, run in priority order
(vLLM zentorch INT8 and BF16 first).

## Stage 2: partitioning one socket (fixed 96 cores)

Question: for a full socket, which split into instances gives the most total throughput?

| Instances | Cores each | CCDs each |
|---|---|---|
| 1 | 96 | 12 |
| 2 | 48 | 6 |
| 3 | 32 | 4 |
| 4 | 24 | 3 |
| 6 | 16 | 2 |
| 12 | 8 | 1 |

- All instances run at once with the offline request pool split evenly.
- Per-instance `max_num_seqs` scaled so the socket's total concurrency stays comparable (e.g. 128 total → 64 × 2,
  32 × 4, ...), plus one run with each instance saturated.
- Engines and precisions: the top candidates from stage 1 (at least vLLM zentorch, vLLM stock, llama.cpp ZenDNN,
  INT8 and BF16).

Output: total socket throughput vs instance count. Also watch latency: more instances means smaller batches per
instance, which usually lowers TPOT but may raise TTFT under load.

## Stage 3: placement within a partition

Question: does the core layout matter for the best split(s) from stage 2?

For the best one or two splits (for example 2 × 48):

- **Contiguous CCD blocks** (instance A = cores 0-47, B = 48-95), the default.
- **Interleaved cores** (A = even cores, B = odd cores), AMD's blog layout. Each instance spans every CCD and
  shares every L3 with the other instance.
- **SMT**: same split, each instance also given its SMT siblings (e.g. A = 0-47 + 192-239), with OpenMP threads set
  to physical cores and to all threads.

Output: which layout to use. Expectation: contiguous CCD blocks win for L3 locality; SMT helps little for
bandwidth-bound decode. Measure rather than assume.

## Stage 4: load sweep on the winners (goodput)

Question: which configuration serves the most requests per second while meeting a latency target?

- Take the best 2-3 configurations from stages 2 and 3 (e.g. 1 × 96, 2 × 48, 4 × 24).
- Online load: sweep the arrival rate from light to overload, in about 8 steps.
- Latency targets (adjust to the product): interactive = p99 TTFT ≤ 2 s and p99 TPOT ≤ 100 ms;
  relaxed = p99 TTFT ≤ 10 s.

Output: goodput (highest request rate meeting the target) per configuration. The peak-throughput winner from
stage 2 and the goodput winner often differ: fewer, larger instances batch better; more, smaller instances queue
less.

## Stage 5: both sockets

Question: does the per-socket result carry over to the whole host?

- **Independent**: the stage 4 winner replicated on socket 1 (cores 96-191, memory node 1). Expect about 2× one
  socket; less means cross-socket interference (shared I/O, the client, or the k3s containers).
- **One instance across both sockets** with vLLM tensor parallel 2 (`--tensor-parallel-size 2`,
  `VLLM_CPU_OMP_THREADS_BIND="0-95|96-191"`), as a comparison only. Expect it to lose to independent instances
  because of cross-socket communication.
- **One instance on 192 cores without tensor parallel**: a sanity check only; expect poor results from remote memory
  access.

## Stage 6: ZenDNN value at the best configuration

Re-run the final configuration with and without ZenDNN on each engine (zentorch vs stock vLLM, ZenDNN vs plain
llama.cpp) using the same partition. Also run AMD's comparison for context: 2 zentorch instances vs 1 stock
instance on the same socket. This separates "multi-instance gain" from "zentorch gain", which AMD's 1.68x headline
mixes together.

## Decision rules

- Pick the configuration with the highest goodput at the target latency. Use peak offline throughput only for pure
  batch jobs.
- Prefer fewer instances when goodput is within 5%: fewer processes are simpler to operate and use less memory.
- If stage 1 shows near-linear scaling to 96 cores, skip to stage 4 with 1 × 96 vs 2 × 48 only.
- Choose per workload if the winners differ (e.g. prefill-heavy traffic may favour one large instance, decode-heavy
  traffic more instances).

## Tooling needed

- `scripts/multi_instance.sh` (to write): launch N instances from a partition spec such as
  `PARTITION="0-47:0 48-95:0"` (cores:mem-node), one port each; wait for readiness; tear down by PID.
  Generalises `scripts/amd_repro_throughput.sh`, which only does 1 vs 2 (even/odd) instances and has a relative
  `OUT` bug.
- Offline driver: split a request pool across instances and report the combined wall time. Either run
  `scripts/offline_batch.py` once per instance in parallel (one process per partition), or send requests over
  HTTP to N servers.
- Online driver: extend `scripts/smoke_client.py` with Poisson arrivals at a fixed rate and round-robin over
  several base URLs.
- Reporting: one table per stage (throughput, per-core throughput, p50/p99 TTFT and TPOT), and charts of
  throughput vs cores (stage 1), vs instance count (stage 2), and goodput vs request rate (stage 4).
- Optional: memory bandwidth per socket (e.g. AMD uProf or `perf` uncore counters) to confirm when decode hits the
  bandwidth limit.

## Rough time budget

| Stage | Points (priority subset) | Time |
|---|---|---|
| 1 Core scaling | ~40 | 5-7 h |
| 2 Partitioning | ~36 | 5-6 h |
| 3 Placement | ~12 | 2 h |
| 4 Load sweep | ~24 | 4-5 h |
| 5 Both sockets | ~6 | 1-2 h |
| 6 ZenDNN value | ~8 | 1-2 h |

About 20-25 hours for the priority subset (vLLM zentorch and stock, llama.cpp ZenDNN; INT8 first). The full grid is
several times that; cut it down using the stage 1 results.
