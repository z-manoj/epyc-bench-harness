# Client requirements: serving benchmarks on AMD EPYC

Suggested values below are for confirmation. Each model is run at **F16** and **Q8**.

All runs are pinned to **32 cores** (4 CCDs). On this host, a 1024-token prompt on 96 cores takes about 1 s (Llama 3.1 8B, BF16, one request). Prefill scales roughly with core count, so the same prompt on 32 cores takes about 3 s. Decode is limited by how fast the 32 cores can read weights from DRAM, and that bandwidth is shared by every user generating at once.

The offline job uses the 20 prompts in each of `prompts/prompts_1024.txt`, `prompts_2048.txt`, and `prompts_4096.txt`, with 128 output tokens. Batch size is 20, so the whole job is one batch. The score is total tokens per second.

## 1. Online serving

**TTFT** is time to first token. **TPS** is output tokens per second for each user. Targets are p95, at 1024-token prompts and 256-token outputs, for **2 concurrent users**.

| Model (Hugging Face ID) | TTFT target (seconds) | TTFT percentile | TPS target per user |
|---|---:|---|---:|
| meta-llama/Llama-3.1-8B-Instruct | 6 | p95 | 8 |
| Qwen/Qwen2-7B-Instruct | 6 | p95 | 8 |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | 10 | p95 | 4 |

Llama 3.1 8B and Qwen2 7B are dense. A single request at F16 is already about 3 s to first token on 32 cores, so 6 s leaves room for a second user. Per-user decode of 8 tokens/s is realistic at Q8 with two users; F16 will be lower because both users share one weight stream.

Mixtral 8x7B is the MoE from AMD’s ZenDNN 5.2 runs: about 47B parameters, 2 of 8 experts active (about 13B per token). vLLM and llama.cpp both run it. On 32 cores the unloaded 1024-token TTFT is about 5 s at F16, and two users share a heavier weight read, so the targets are 10 s and 4 tokens/s per user. Weights are about 94 GB at F16 and 47 GB at Q8, which fits on this host.

## 2. Offline serving (batched jobs)

| Requirement | Answer |
|---|---|
| Models (Hugging Face IDs) | meta-llama/Llama-3.1-8B-Instruct; Qwen/Qwen2-7B-Instruct; mistralai/Mixtral-8x7B-Instruct-v0.1 |
| Precision / quantisation for each model | F16 and Q8 |
| Input lengths (tokens per request) | 1024, 2048, and 4096 |
| Output length (tokens per request) | 128 |
| Number of requests per job | 20 (one pass over prompts/prompts_{1024,2048,4096}.txt) |
| Batch size | 20 |
| Maximum batch / job completion time, if any | None |
| Primary target | Total TPS |

Batch size equals the job size, so all 20 prompts are in flight together. No completion-time cap: the run is scored on total output tokens per second.
