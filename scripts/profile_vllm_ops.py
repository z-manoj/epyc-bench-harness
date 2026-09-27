"""Profile which ops vLLM actually runs for one 1024-token prefill + short decode (in-process engine).

Run inside a vLLM env with VLLM_ENABLE_V1_MULTIPROCESSING=0 so the worker runs in this process.
"""
import argparse
import random

import torch
from torch.profiler import ProfilerActivity, profile
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def prompt(seed, n=1024):
    rng = random.Random(seed)
    return TokensPrompt(prompt_token_ids=[128000] + [rng.randrange(1000, 127000) for _ in range(n - 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096, enable_prefix_caching=False)
    sp = SamplingParams(max_tokens=args.gen, ignore_eos=True, temperature=0.0)
    llm.generate([prompt(1)], sp, use_tqdm=False)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        llm.generate([prompt(2)], sp, use_tqdm=False)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.top, max_name_column_width=70))
    try:
        from zentorch._utils import counters
        print("zentorch counters:", {k: dict(v) for k, v in counters.items() if v})
    except ImportError:
        print("zentorch not installed")


if __name__ == "__main__":
    main()
