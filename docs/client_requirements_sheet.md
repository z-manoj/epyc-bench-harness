# Client requirements sheet: Phase 3(c) production deployment on EPYC

Please fill in the **Answer** column. Where an answer is unknown, leave it blank and we will use the **Default**
shown; you can change it later, but changes after experiments start may require re-runs.

**Priority**: **P0** = needed before experiments can start; **P1** = needed within the first 4 weeks;
**P2** = needed before the final report.

Context: this sheet supports the Phase 3(c) deliverable (Kubernetes deployment guide, single- vs multi-instance
decision framework, load-balancer reference, sustained 8-16 users per 96-core node, vLLM vs llama.cpp runtime
benchmark). Plan: [phase3c_production_deployment_plan.md](phase3c_production_deployment_plan.md).

## Minimum to start (P0 summary)

1. The three workload types, with prompt/output lengths (section B).
2. Latency targets per workload (section C).
3. The exact 7B-class and 14B-class models, and permitted precisions (section D).
4. Target hardware and what "96-core node" means (section E).
5. Kubernetes environment and our level of access (section F).
6. Phase 3 start date and review contacts (section A).

---

## A. Project and contacts

| ID | Question | Why we need it | Default | Priority | Answer |
|---|---|---|---|---|---|
| A1 | Phase 3 start date (the due date is 5 months later) | Fixes the timeline and checkpoints | Date this sheet is returned | P0 | |
| A2 | Primary reviewer and technical contact(s) at AMD | Checkpoint reviews and questions | Sumit (reviewer) | P0 | |
| A3 | Review cadence and checkpoint dates | Plan proposes checkpoints at ~week 7 and ~week 15 | Bi-weekly sync, 2 checkpoints | P1 | |
| A4 | Preferred communication channel and escalation path | Blocking issues | Email + shared chat channel | P1 | |

## B. Workloads (the three playbook workload types)

Please fill one column per workload. If the playbook already defines them, a link is enough.

| ID | Question | Why we need it | Default W1 / W2 / W3 | Priority | W1 answer | W2 answer | W3 answer |
|---|---|---|---|---|---|---|---|
| B1 | Name and use case | Maps results to business scenarios | Interactive chat / Long-context RAG / Batch offline | P0 | | | |
| B2 | Prompt length in tokens: median, p95, max | Prompt processing dominates latency on CPU | 1K, 2K, 4K / 8K, 8K, 16K / 1K, 2K, 4K | P0 | | | |
| B3 | Output length in tokens: median, p95, max | Drives generation time and throughput | 300, 600, 1K / 256, 512, 1K / 128, 256, 512 | P0 | | | |
| B4 | Share of each prompt that is common across requests (system prompt, reused documents), in tokens or % | Prefix caching can cut latency several-fold | 500 tokens / 50% / 0% | P0 | | | |
| B5 | Multi-turn conversations? Typical number of turns | Session affinity and cache reuse | Yes, 3-8 / Yes, 2-4 / No | P1 | | | |
| B6 | User think time between turns | Defines how a "user" loads the system | 10-30 s / 30-60 s / n.a. | P1 | | | |
| B7 | Expected peak request rate or peak concurrent users per node | Sizing and autoscaling targets | 8-16 users / 8-16 users / n.a. | P0 | | | |
| B8 | Traffic pattern: steady, daily peaks, bursts (size and duration) | Burst and autoscaling tests | Steady with 3× 30 s bursts | P1 | | | |
| B9 | Streaming responses required? | Gateway and metrics setup | Yes / Yes / No | P1 | | | |
| B10 | Can you provide representative prompts or a dataset (anonymised is fine)? | Realistic lengths and prefix sharing | We generate synthetic prompts matching B2-B4 | P1 | | | |
| B11 | Sampling settings (temperature, top-p, max tokens, stop sequences) | Affects output length and speed | temperature 0.7, top-p 0.95 | P2 | | | |
| B12 | For batch (W3): job size, deadline or throughput target | Defines the W3 success metric | Maximise tokens/s | P1 | | | |

## C. Service targets (what "a user is served" means)

| ID | Question | Why we need it | Default | Priority | Answer |
|---|---|---|---|---|---|
| C1 | Time to first token target per workload, and percentile (p95 or p99) | Defines capacity; the 8-16 user target is judged against it | W1: p95 ≤ 3 s (1K prompt); W2: p95 ≤ 15 s (8K uncached); W3: none | P0 | |
| C2 | Time per output token (streaming speed) target | Second half of user experience | p95 ≤ 100 ms (≥ 10 tokens/s per user) | P0 | |
| C3 | End-to-end response time target, if any | Some products use total time instead | None | P1 | |
| C4 | Acceptable error / timeout rate | Pass criterion for soak tests | < 0.1% | P1 | |
| C5 | Is a hard TTFT guarantee needed for some users (premium tier), or are percentiles enough? | Decides whether to evaluate a reserved-capacity tier | Percentiles enough | P1 | |
| C6 | Is a relaxed tier acceptable where interactive targets cannot be met (e.g. 14B with 8K prompts)? | 16 users on W2 with 14B is at risk on one socket | Yes, report both tiers | P1 | |
| C7 | Definition of "sustained load": duration and pass rule | Soak test design | ≥ 1 h per point, 4 h soak at 16 users; targets met in every 5-minute window | P0 | |
| C8 | Is 8-16 users per node a minimum to prove, or a range to characterise? | Report framing | Characterise; report max users meeting targets | P1 | |

## D. Models

| ID | Question | Why we need it | Default | Priority | Answer |
|---|---|---|---|---|---|
| D1 | Exact 7B-8B class model (Hugging Face ID) | Benchmark subject | meta-llama/Llama-3.1-8B-Instruct | P0 | |
| D2 | Exact 14B class model (Hugging Face ID) | Benchmark subject | Qwen/Qwen2.5-14B-Instruct | P0 | |
| D3 | Permitted precisions: BF16, INT8 (W8A8 / Q8_0), INT4 (e.g. Q4_K_M, AWQ)? | INT8 is ~1.6× faster; quality trade-off is your call | BF16 and INT8 | P0 | |
| D4 | Quality constraints for quantised models (accuracy tolerance, eval to use) | Whether INT8 / INT4 results are admissible | INT8 accepted without re-evaluation | P1 | |
| D5 | Maximum context length to support | KV cache sizing and memory requests | 16K tokens | P1 | |
| D6 | Model licence / gated-access approval in place (e.g. Meta, Qwen) | Download and redistribution in images | We use our own HF access | P0 | |
| D7 | May model weights be baked into container images, or must they be mounted? | Image and storage design | Mounted from node-local storage | P2 | |

## E. Hardware

| ID | Question | Why we need it | Default | Priority | Answer |
|---|---|---|---|---|---|
| E1 | Target EPYC SKU for the guide (e.g. 9R14 Genoa, 9755 Turin) | Results differ by generation; AMD's published numbers use Turin | 2× EPYC 9R14 (current test host) | P0 | |
| E2 | Meaning of "96-core node": one socket of a 2-socket server, or a single-socket 96-core server? | Unit of the 8-16 user target | One socket (one NUMA node) | P0 | |
| E3 | Hardware we can use: number of nodes, dedicated or shared, for how long | Multi-node and autoscaling tests; measurement noise | 1 host, shared (another team's k3s pods present) | P0 | |
| E4 | Memory per node and DIMM population (channels, speed) | Decode speed is memory-bandwidth bound | 1.5 TB, as installed | P1 | |
| E5 | BIOS settings we may change or you require: NPS (1/2/4), SMT, power / determinism profile, boost | NUMA layout changes pod sizing | NPS1, SMT on, current profile; no BIOS changes | P1 | |
| E6 | OS and kernel for the guide | Reproducibility | Ubuntu 22.04 (current) | P2 | |
| E7 | Cloud equivalent instance type, if cloud deployment is also in scope | Cost model and portability notes | None | P2 | |

## F. Kubernetes environment

| ID | Question | Why we need it | Default | Priority | Answer |
|---|---|---|---|---|---|
| F1 | Distribution and version (upstream, k3s, RKE2, OpenShift, EKS, ...) | Available kubelet features (e.g. CPU manager options) | Upstream 1.31+ | P0 | |
| F2 | Our access level: cluster admin, namespace only, or none (you apply our manifests)? | NUMA-aware setup needs kubelet configuration changes | Cluster admin on a test cluster | P0 | |
| F3 | May we change kubelet config (CPU manager static, topology manager, memory manager, reserved CPUs) and restart kubelet? | Core of the NUMA-aware guide | Yes, on dedicated test nodes | P0 | |
| F4 | cgroup version and driver (v2 + systemd?) | Required for the guide | cgroup v2, systemd | P1 | |
| F5 | Container runtime and image registry we should use | Image build and pull | containerd; our registry | P1 | |
| F6 | Monitoring stack available (Prometheus, Grafana), and may we install prometheus-adapter or KEDA? | Autoscaling on queue depth / TTFT | Prometheus + KEDA installable | P1 | |
| F7 | Preferred ingress / gateway (Envoy Gateway, NGINX, Istio, Gateway API implementation, other) | Load-balancer reference config | Envoy Gateway, with NGINX notes | P1 | |
| F8 | Packaging preference (Helm, kustomize, operator) and GitOps tooling | Deliverable format | Helm chart | P2 | |
| F9 | Security constraints: non-root, read-only root filesystem, pod security level, no internet egress, image scanning | Pod spec and model download design | Non-root, restricted where possible, egress allowed | P1 | |
| F10 | Storage for model weights (local disk, NFS, object store, PVC class) | Startup time and scaling speed | Node-local disk | P1 | |

## G. Software constraints

| ID | Question | Why we need it | Default | Priority | Answer |
|---|---|---|---|---|---|
| G1 | Required or pinned versions: vLLM, llama.cpp, PyTorch | Reproducibility and support | Latest stable that runs on the target OS (vLLM 0.28, llama.cpp current) | P1 | |
| G2 | ZenDNN / zentorch versions to use; access to AMD internal builds or the private AMD-Zenai repositories | Our current zentorch is built from the public repository | Public zentorch release | P1 | |
| G3 | Any runtimes besides vLLM and llama.cpp to include? (Ollama is out of scope) | Runtime matrix scope | vLLM and llama.cpp only | P0 | |
| G4 | Licensing or open-source constraints on deliverables (can scripts and configs be published?) | Repository and report handling | Shared privately with AMD | P2 | |

## H. Cost model

| ID | Question | Why we need it | Default | Priority | Answer |
|---|---|---|---|---|---|
| H1 | Cost basis for cost per request: cloud on-demand price, reserved price, or on-prem TCO | Cost/request = node cost per second ÷ requests per second sustained within targets | Public cloud on-demand price for an equivalent Genoa 96-core instance | P1 | |
| H2 | If on-prem: $/hour per node (or capex, lifetime, power, and facility cost to derive it) | Same | Not included | P2 | |
| H3 | Currency, and whether power is included | Report consistency | USD, power included in cloud price | P2 | |
| H4 | Comparison baseline for cost (e.g. a GPU instance price) wanted? | Enterprise decision context | None | P2 | |

## I. Deliverables and acceptance

| ID | Question | Why we need it | Default | Priority | Answer |
|---|---|---|---|---|---|
| I1 | Report format (Markdown in repo, PDF, slides) | Deliverable production | Markdown + PDF export; summary slides for the demo | P1 | |
| I2 | Demo format for acceptance (live on the cluster, recorded, report walkthrough) | Demo preparation | Live demo on the test cluster + report walkthrough | P1 | |
| I3 | Audience of the decision matrix (AMD internal, enterprise customers, public) | Level of detail and confidentiality | Enterprise practitioners, AMD-reviewed | P1 | |
| I4 | Any results or configurations that must not be published | Confidentiality | All results confidential until AMD approves | P1 | |
| I5 | Additional acceptance criteria beyond the three listed | Avoid late surprises | None | P1 | |

## J. Access and approvals

| ID | Question | Why we need it | Default | Priority | Answer |
|---|---|---|---|---|---|
| J1 | Remote access to the test nodes / cluster (VPN, SSH, bastion) | Running experiments | Current SSH access | P0 | |
| J2 | Hugging Face access for gated models (org token or approval on our accounts) | Model downloads | Our own token | P0 | |
| J3 | GitHub access to AMD-Zenai repositories (if G2 requires internal builds) | zentorch / ZenDNN internal versions | Not needed | P1 | |
| J4 | Maintenance windows or times the hardware is unavailable; other workloads sharing it | Measurement scheduling and noise | Available 24/7, shared | P1 | |
| J5 | Approval to run long soak tests (4 h+) at full load | Sustained load validation | Approved | P1 | |
