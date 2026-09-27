"""Offline batch test: one LLM.generate() call over N real-text prompts of exactly PROMPT_LEN tokens.

Runs in whichever vLLM env is active (zentorch is picked up automatically if installed). Writes one JSON line per run.
"""
import argparse
import json
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def build_prompts(tok, path, n, prompt_len):
    texts = [json.loads(line)["text"] for line in open(path) if line.strip()]
    prompts = []
    for i in range(n):
        ids = tok(texts[i % len(texts)])["input_ids"]
        j = i + 1
        while len(ids) < prompt_len:
            ids += tok(texts[j % len(texts)], add_special_tokens=False)["input_ids"]
            j += 1
        prompts.append(TokensPrompt(prompt_token_ids=ids[:prompt_len]))
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-chunked-prefill", action="store_true")
    args = ap.parse_args()

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.prompt_len + args.gen + 64,
              enable_prefix_caching=False, max_num_seqs=max(args.n, 1),
              enable_chunked_prefill=not args.no_chunked_prefill)
    print(f"[{args.label}] chunked_prefill={llm.llm_engine.vllm_config.scheduler_config.enable_chunked_prefill} "
          f"max_num_batched_tokens={llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens}", flush=True)
    prompts = build_prompts(llm.get_tokenizer(), args.prompts, args.n, args.prompt_len)
    # per-request seed is unsupported on vLLM CPU ("CPU Generator does not use offset")
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=args.gen, ignore_eos=True)

    llm.generate(prompts[:2], sp, use_tqdm=False)
    for run in range(1, args.runs + 1):
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp, use_tqdm=False)
        wall = time.perf_counter() - t0
        n_in = sum(len(o.prompt_token_ids) for o in outputs)
        n_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        rec = {"label": args.label, "run": run, "n": args.n, "prompt_len": args.prompt_len, "gen_len": args.gen,
               "wall_s": wall, "prompt_tokens": n_in, "output_tokens": n_out,
               "output_tok_s": n_out / wall, "total_tok_s": (n_in + n_out) / wall, "req_s": args.n / wall}
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{args.label}] run {run}: {args.n} x {args.prompt_len} in / {args.gen} out, wall={wall:.2f}s "
              f"output={rec['output_tok_s']:.1f} tok/s total={rec['total_tok_s']:.1f} tok/s "
              f"(in={n_in} out={n_out})", flush=True)
    print(f"[{args.label}] sample output: {outputs[0].outputs[0].text[:200]!r}", flush=True)


if __name__ == "__main__":
    main()
