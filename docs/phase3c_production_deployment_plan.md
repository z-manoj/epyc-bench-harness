# Phase 3(c) plan: production deployment strategies on EPYC

Plan for the Phase 3(c) deliverable: Kubernetes orchestration and topology guidance, a single- vs multi-instance
decision framework, sustained multi-user validation, and a runtime selection benchmark. Ollama is out of scope;
the runtime comparison is **vLLM (CPU backend, with and without zentorch) vs llama.cpp server (with and without
ZenDNN)**.

Due: 5 months from the start of Phase 3. Reviewer: AMD (Sumit).

Related plans in this repo, reused here rather than repeated:
- [throughput_experiment_plan.md](throughput_experiment_plan.md): instance split at saturation (batch jobs).
- [multi_user_experiment_plan.md](multi_user_experiment_plan.md): prefill/decode cost map, capacity at latency
  targets, shared vs guaranteed serving.
- [deployment_recommendations.md](deployment_recommendations.md): current measured recommendations.

## 1. Acceptance criteria and what proves each

| # | Criterion | Evidence delivered |
|---|---|---|
| 1 | NUMA-aware Kubernetes deployment guide and single- vs multi-instance decision framework | Deployment guide (node config, pod specs, manifests / Helm chart); decision framework document backed by the cost map and topology experiments; demo of pods pinned to one NUMA node with exclusive cores |
| 2 | Load-balancer configuration reference and sustained concurrency results (8-16 users per 96-core node) | Gateway configs (least-request, session affinity) with measured effect; soak-test report: 8, 12 and 16 users per 96-core node for each workload type and model size, sustained ≥ 1 h within targets |
| 3 | vLLM vs llama.cpp runtime benchmark report and decision matrix | Report with TTFT, tokens/s and cost/request per runtime, workload and model size; one-page decision matrix for practitioners |

## 2. Assumptions to confirm early

These change the experiment matrix; defaults are used until confirmed.

| Item | Default assumption |
|---|---|
| The three playbook workload types | **W1 interactive chat** (prompts ~0.5-2K tokens, answers ~200-400), **W2 long-context / RAG** (prompts ~8K, answers ~256), **W3 batch / offline** (throughput only, no latency target) |
| "96-core EPYC node" | One 96-core socket (one NUMA node) of the 2× EPYC 9R14 host. Results are also reported per full host (2 sockets) |
| Models (7B-14B) | Llama 3.1 8B Instruct (have it) and one 14B model, e.g. Qwen2.5-14B-Instruct or Phi-4 (14B); BF16 and INT8 for each |
| Service targets for "users served" | Interactive tier from the multi-user plan: p95 TTFT within a prompt-length-scaled budget, p95 TPOT ≤ 100 ms, errors < 0.1%. W2 targets are scaled to the ~9 s single-request floor for 8K prompts |
| User model | Closed loop: each user sends a request, reads the streamed answer, thinks 10-30 s (W1) or 30-60 s (W2), repeats; multi-turn conversations of 3-8 turns |
| Cost basis | $/hour for a 96-core node, as a parameter: a public cloud on-demand price for an equivalent Genoa instance, plus an optional on-prem TCO figure supplied by AMD |
| Kubernetes | A recent upstream version (1.31+) on this host or an equivalent EPYC node, cgroup v2, containerd, with kubelet configuration under our control |

Risk to flag now: **16 users on W2 with a 14B model is unlikely to meet interactive targets on one 96-core socket**
without high prefix-cache hit rates. An uncached 8K prompt already takes ~9 s with the 8B model on a full socket
(W8A8), about 6-7 new prompts per minute; a 14B model roughly halves that. The concurrency report will state the
supported users per workload and model honestly, with the conditions (cache hit rate, target tier) under which
8-16 is met.

## 3. Workstreams

### WS1: Kubernetes deployment on EPYC (criterion 1)

**Node configuration** (kubelet config, documented and shipped as a config file):
- cgroup v2 with the systemd cgroup driver (containerd and kubelet both `systemd`).
- CPU manager `static`, with policy options:
  - `full-pcpus-only: "true"`: allocate whole physical cores (both SMT siblings), so two pods never share a core.
  - `prefer-align-cpus-by-uncorecache` where the Kubernetes version supports it: packs a pod's CPUs onto as few
    L3 domains (CCDs) as possible. Otherwise document the CCD layout and size pods in multiples of 8 cores.
- Topology manager `single-numa-node`, scope `pod`: a pod's CPUs and memory come from one NUMA node, or the pod is
  rejected rather than silently spanning sockets.
- Memory manager `Static` so guaranteed pods get memory from the same NUMA node as their CPUs.
- `reservedSystemCPUs`: one physical core (2 logical CPUs) per NUMA node for the OS and kubelet, e.g. `0,192` and
  `96,288`, plus `kubeReserved` / `systemReserved` memory.
- Optional: 2 MB huge pages for model weights (measure; not assumed to help).

**Scheduling across NUMA nodes**: the default scheduler does not see NUMA, so a pod can land on a node where no
single NUMA node has room and then fail admission (`TopologyAffinityError`). Use the topology-aware scheduler
plugin with NodeResourceTopology (via the Node Feature Discovery / resource-topology exporter), or, for a fixed
fleet, one pod size that divides a NUMA node exactly.

**Inference pod spec** (Guaranteed QoS):
- `requests == limits`, integer CPUs, in multiples of 2 (one physical core = 2 logical CPUs with SMT).
- Sizes calibrated to a 96-core socket after reserving one core: 1 pod × 94 cores (188 CPUs), 2 × 47, 4 × 23;
  or, if CCD alignment matters more than using every core, 2 × 40 / 4 × 16 / 6 × 16 on whole CCDs. The topology
  experiments (WS2) pick the sizes.
- Memory request = weights + KV cache + runtime overhead (e.g. 8B BF16 ≈ 16 GB weights + KV cache sized for the
  target users × context + ~4 GB; add ~8 GB for llama.cpp ZenDNN's extra weight copy). Measured per runtime.
- Container entrypoint reads its assigned CPU set from `/sys/fs/cgroup/cpuset.cpus.effective` and derives thread
  binding from it: one thread per physical core (`VLLM_CPU_OMP_THREADS_BIND` for vLLM; `-t` and CPU mask for
  llama.cpp). Nothing hard-codes core IDs.
- Same runtime environment as bare metal: `LD_PRELOAD` of libiomp5 + tcmalloc, `TORCHINDUCTOR_FREEZING` per
  runtime, KV cache size from the memory request.
- Probes: `startupProbe` long enough for model load plus compile warmup (1-3 min), `readinessProbe` on the health
  endpoint after a warmup request, so a pod receives traffic only when warm.
- Model weights from a node-local volume or a read-only PVC pre-populated per node, not pulled at start.

**Autoscaling (HPA on serving signals, not CPU)**: CPU utilisation is useless here: a pinned inference pod shows
high CPU whether it is keeping up or drowning.
- Metrics: vLLM exposes Prometheus metrics (`vllm:num_requests_waiting`, `vllm:num_requests_running`,
  `vllm:time_to_first_token_seconds` histogram); llama.cpp with `--metrics` exposes processing / deferred requests.
  The gateway also measures TTFT per request.
- Pipeline: Prometheus → prometheus-adapter (custom metrics API) or KEDA.
- Scale-up triggers: waiting requests per pod above a threshold (primary, fast) or p95 TTFT over 2-5 minutes
  above target (secondary, catches slow degradation).
- Behaviour: fast scale-up with a short stabilisation window; slow scale-down (e.g. 10 minutes) because a new pod
  takes minutes to become ready; `minReplicas` sized for baseline load so scaling covers peaks only.
- On a single host the HPA can only scale between NUMA-sized slots (e.g. 1 → 2 → 4 pods). The HPA test shows the
  mechanism and its timing; multi-node scaling is described but only tested if a second node is available.

**Artifacts**: kubelet config, Helm chart (or kustomize) for vLLM and llama.cpp with the pod sizes as values,
Prometheus scrape config and recording rules, HPA / KEDA specs, gateway configs (WS3), and a validation checklist
(verify exclusive cpusets, NUMA-local memory, and thread binding inside a running pod).

**Validation**:
- Kubernetes overhead: the same instance on bare metal (numactl) vs in a Guaranteed pod on the same cores; target
  within 3% on TTFT and throughput. Any larger gap is a configuration bug to fix before other results count.
- Negative test: the same pod without the static CPU manager / topology manager, showing the latency variance and
  cross-NUMA penalty the configuration avoids.

### WS2: single- vs multi-instance decision framework (criterion 1)

Built from measurements, not rules of thumb:

1. **Cost map** (multi-user plan, stage 1): TTFT, prompt tokens/s and TPOT for prompt lengths 512-8192 on
   instances of 4-96 cores, isolated and with neighbours busy. Gives per-core efficiency vs instance size.
2. **Topology comparison under load**: 1 × 96, 2 × 48, 3 × 32, 4 × 24 cores per socket, all with continuous
   batching and each tuned for its own batch limit, on W1, W2 and W3, for each runtime and model size. Metrics:
   capacity within targets (W1, W2) and peak throughput (W3).
3. **Guaranteed tier** (optional): one request per instance with admission control, for products needing a hard
   TTFT promise.

The framework document turns the results into a decision tree, expected to look like:
- Prompt-heavy traffic (W2) or strict TTFT → fewer, larger instances (one per NUMA node).
- Decode-heavy, many concurrent short chats (W1) → more instances if the cost map shows higher per-core
  efficiency for smaller instances.
- Batch (W3) → whatever maximises throughput at saturation (throughput plan).
- Larger model (14B) → shifts toward fewer, larger instances (prefill cost doubles).
- Never span NUMA nodes with one instance unless tensor parallel measurably helps TTFT.

Each branch cites the measured numbers behind it.

### WS3: load balancing and multi-user serving (criterion 2)

**Gateway options** (pick one as the reference, document the other):
- Envoy (or an Envoy-based Gateway API implementation) with `LEAST_REQUEST` load balancing, and ring-hash /
  Maglev on a session header for affinity.
- NGINX with `least_conn`, and `hash $http_x_session_id consistent` for affinity.
- Note the Kubernetes Gateway API Inference Extension (queue- and KV-cache-aware endpoint picking) as the direction
  of travel; evaluate if it supports CPU vLLM by then.

**Policies compared** (multi-turn W1 traffic, 2 and 4 pods per node):
- Round robin (baseline).
- Least outstanding requests: avoids piling long prompts onto one pod.
- Session affinity (consistent hash on conversation ID): every turn of a conversation reaches the pod holding its
  prefix cache, so only the new message is processed.
- Affinity with bounded load: affinity unless the target pod is over a load threshold, then least-request. Expected
  best overall; measure against the two above.

Metrics: capacity within targets, TTFT p95, prefix-cache hit rate per pod, load imbalance across pods.

**Streaming configuration**: SSE passthrough with response buffering off, idle timeouts longer than the longest
generation, and connection handling tested with long streams (llama.cpp closes the connection after each
streamed response; the gateway must not treat that as an error).

**Sustained concurrency test (the 8-16 users target)**:
- Per 96-core node (one socket), for each workload (W1, W2), model (8B, 14B), and the best runtime and topology
  from WS2/WS4: closed-loop users at 8, 12 and 16.
- Duration ≥ 1 hour per point after warmup, plus one 4-hour soak at 16 users for the headline configuration.
- Pass: all targets met over the whole window (reported per 5-minute interval, so degradation over time is visible),
  no errors or restarts, memory stable.
- Report the maximum users within targets for each combination, and the conditions (cache hit rate, target tier).
- Load generator runs outside the node under test (or on a reserved NUMA node) and records per-request TTFT, TPOT,
  inter-token gaps and errors.

### WS4: runtime selection benchmark (criterion 3)

**Matrix** (identical hardware, same pinning, same node configuration):

| Dimension | Values |
|---|---|
| Runtime | vLLM + zentorch, vLLM stock, llama.cpp + ZenDNN, llama.cpp plain CPU |
| Model | Llama 3.1 8B, 14B model |
| Precision | BF16; INT8 (W8A8 for vLLM, Q8_0 for llama.cpp) |
| Workload | W1, W2, W3 |
| Load | Single user (latency floor) and the WS3 concurrency levels |

**Metrics**:
- TTFT p50 / p95, TPOT p50 / p95, output tokens/s per user and aggregate.
- Capacity: users or requests/s within targets (W1, W2); peak throughput (W3).
- **Cost per request** = node cost per second ÷ requests per second sustained within targets
  (= $/hour ÷ 3600 ÷ req/s). Also cost per 1M output tokens. Computed at the measured capacity, not at peak, so it
  reflects a real service. Shown for both cost bases (cloud price, on-prem TCO) if available.
- Operational factors (qualitative, scored): startup time, memory footprint, stability under soak, metrics and
  health endpoints, prefix caching, quantisation formats, OpenAI API compatibility, ease of Kubernetes packaging.

**Decision matrix**: one page, rows = workload × model size, columns = runtime; each cell gives TTFT p95,
capacity, cost/request and a recommendation mark, with a short rationale per row. Plus a "choose X when" summary
for practitioners.

Known inputs from current results: vLLM leads on TTFT and batching; llama.cpp + ZenDNN halves TTFT vs plain
llama.cpp and has the best single-stream decode with Q8_0; zentorch gains 3-7% over stock vLLM on batch; stock
vLLM 0.28 crashes with Inductor freezing on (workaround `FREEZING=0`).

## 4. Timeline (about 20 weeks)

| Weeks | Work | Milestone |
|---|---|---|
| 1-2 | Confirm assumptions (workloads, targets, 14B model, cost basis). Download 14B models (BF16, INT8, GGUF). Build container images for all four runtimes. Load generator: closed loop with think time, multi-turn sessions, Poisson mode, multiple endpoints, session header | Images and load generator ready |
| 3-5 | Cost map (bare metal) for 8B then 14B, vLLM zentorch INT8 first; loaded pass | Cost map report |
| 5-7 | Kubernetes node configuration, Helm chart, NUMA and cpuset validation, bare-metal vs pod overhead test, negative test | **Checkpoint 1**: NUMA-aware deployment demo |
| 7-10 | Topology comparison (WS2) under load, per runtime / model / workload; draft decision framework | Decision framework draft |
| 10-13 | Runtime benchmark (WS4) at the chosen topologies; cost model | Runtime report draft |
| 13-15 | Gateway policies (WS3); HPA on queue depth / TTFT (Prometheus, adapter or KEDA), scaling tests | **Checkpoint 2**: LB reference + HPA demo |
| 15-18 | Sustained concurrency tests: 8 / 12 / 16 users, W1 and W2, 8B and 14B, 1 h each; 4 h soak | Concurrency report |
| 18-20 | Final deployment guide, decision framework, LB reference, runtime report and matrix; reproducibility pass; demo to AMD | **Acceptance demo** |

Checkpoints 1 and 2 are natural review points with AMD before the final demo.

## 5. Deliverable documents

1. `docs/k8s_deployment_guide.md`: node configuration, pod sizing, manifests / Helm chart usage, validation
   checklist, autoscaling.
2. `docs/topology_decision_framework.md`: decision tree with the measurements behind each branch.
3. `docs/load_balancer_reference.md`: gateway configs, policy comparison results, streaming settings.
4. `docs/concurrency_validation_report.md`: 8-16 user soak results per workload and model.
5. `docs/runtime_benchmark_report.md`: full results, cost model, decision matrix.
6. `deploy/`: kubelet config, Helm chart, gateway configs, Prometheus / HPA / KEDA specs.
7. `scripts/`: load generator and experiment drivers, so every number is reproducible.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| 16 users on W2 with 14B misses interactive targets | Report supported users honestly per condition; show the prefix-cache and INT8 levers; offer the relaxed tier or a 2-socket node as the configuration that meets it |
| Stock vLLM crashes with freezing on | Run stock with `FREEZING=0`; track upstream fix (vllm#46131) |
| Newer kubelet options (uncore-cache alignment) unavailable | Fall back to CCD-multiple pod sizes; document both |
| Pods fail NUMA admission | Topology-aware scheduling, or fixed pod sizes that tile a NUMA node |
| Kubernetes adds overhead vs bare metal | Overhead test in week 5-7 gates later work; fix configuration before continuing |
| Only one physical host | Treat each socket as a 96-core node; HPA tested between NUMA slots; state multi-node behaviour as design guidance |
| Other workloads on the shared host (k3s Qwen pods) | Dedicated node or cordoned NUMA node for measurements; check idle before each run |
| Workload definitions change | Workloads are parameters of the load generator; matrix re-runs are scripted |
