"""Smoke client: N runs, each a burst of --concurrency simultaneous requests with exact-length token-ID prompts
and a fixed decode length.

Works against llama-server and vLLM through /v1/completions (streaming). Appends one JSON line per measured request;
run_wall_s is the burst's start-to-last-finish time.
"""
import argparse
import asyncio
import json
import random
import time

import aiohttp

# Defaults are Llama 3: BOS 128000, ordinary-vocab ids below 128000. Mixtral: --bos 1 --vocab-hi 31000.
# Qwen2 has no BOS (--bos -1).
def make_prompt(args, seed: int) -> list[int]:
    rng = random.Random(seed)
    bos = [args.bos] if args.bos >= 0 else []
    return bos + [rng.randrange(args.vocab_lo, args.vocab_hi) for _ in range(args.prompt_len - len(bos))]


async def one_request(session, args, seed):
    body = {
        "model": args.model,
        "prompt": make_prompt(args, seed),
        "max_tokens": args.gen_len,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    if args.engine == "llamacpp":
        body["cache_prompt"] = False
    t0 = time.perf_counter()
    t0_epoch = time.time()
    ttft = None
    token_times = []
    usage, timings = None, None
    async with session.post(f"{args.url}/v1/completions", json=body) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:500]}")
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            now = time.perf_counter()
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("timings"):
                timings = chunk["timings"]
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                if ttft is None:
                    ttft = now - t0
                token_times.append(now)
    e2e = time.perf_counter() - t0
    n_out = (usage or {}).get("completion_tokens") or len(token_times)
    n_in = (usage or {}).get("prompt_tokens")
    tpot = (e2e - ttft) / (n_out - 1) if n_out > 1 else float("nan")
    rec = {
        "ttft_s": ttft,
        "tpot_ms": tpot * 1000,
        "e2e_s": e2e,
        "prompt_tokens": n_in,
        "completion_tokens": n_out,
        "stream_chunks": len(token_times),
        "prefill_tok_s": args.prompt_len / ttft,
        "decode_tok_s": 1.0 / tpot,
        "t_start": t0_epoch,
        "t_first": t0_epoch + ttft,
        "t_end": t0_epoch + e2e,
    }
    if timings:
        rec["server_prompt_ms"] = timings.get("prompt_ms")
        rec["server_predicted_ms"] = timings.get("predicted_ms")
    return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--engine", choices=["llamacpp", "vllm"], required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="llama")
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--scenario", default="")
    ap.add_argument("--bos", type=int, default=128000)
    ap.add_argument("--vocab-lo", type=int, default=1000)
    ap.add_argument("--vocab-hi", type=int, default=127000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    timeout = aiohttp.ClientTimeout(total=3600)
    connector = aiohttp.TCPConnector(force_close=True, limit=0)
    c = args.concurrency
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for i in range(args.warmup):
            r = await one_request(session, args, seed=10_000 + i)
            print(f"[{args.label}] warmup {i}: ttft={r['ttft_s']:.2f}s tpot={r['tpot_ms']:.1f}ms", flush=True)
        for run in range(1, args.runs + 1):
            t0 = time.perf_counter()
            recs = await asyncio.gather(*[one_request(session, args, seed=run * 100_000 + args.prompt_len + i)
                                          for i in range(c)])
            wall = time.perf_counter() - t0
            with open(args.out, "a") as f:
                for i, r in enumerate(recs):
                    r.update(label=args.label, engine=args.engine, run=run, req=i, concurrency=c,
                             scenario=args.scenario, prompt_len=args.prompt_len, gen_len=args.gen_len,
                             run_wall_s=wall)
                    if r["completion_tokens"] != args.gen_len or (
                            r["prompt_tokens"] and r["prompt_tokens"] != args.prompt_len):
                        r["warning"] = "token count mismatch"
                    f.write(json.dumps(r) + "\n")
            bad = sum("warning" in r for r in recs)
            mean = lambda k: sum(r[k] for r in recs) / c  # noqa: E731
            print(f"[{args.label}] run {run}: c={c} ttft={mean('ttft_s'):.2f}s tpot={mean('tpot_ms'):.1f}ms "
                  f"e2e={mean('e2e_s'):.2f}s wall={wall:.2f}s out_tok/s={c * args.gen_len / wall:.1f} "
                  f"in={recs[0]['prompt_tokens']} out={recs[0]['completion_tokens']}"
                  f"{f' WARN {bad} token mismatches' if bad else ''}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
