"""Mock OpenAI-compatible streaming completion server for harness self-tests.

prompt_tokens = number of whitespace-separated words in the prompt.  Decode speed degrades mildly
with the number of active requests to mimic batching.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from aiohttp import web


def build_app(a: argparse.Namespace) -> web.Application:
    state = {"ready_at": time.monotonic() + a.startup_delay_s, "active": 0, "served": 0}
    slots = asyncio.Semaphore(a.parallel)

    async def health(_: web.Request) -> web.Response:
        if time.monotonic() < state["ready_at"]:
            return web.json_response({"status": "loading"}, status=503)
        return web.json_response({"status": "ok"})

    async def models(_: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": a.model, "object": "model"}]})

    def maybe_crash() -> None:
        if a.crash_after and state["served"] >= a.crash_after:
            marker = Path(a.crash_once_file) if a.crash_once_file else None
            if marker is None or not marker.exists():
                if marker is not None:
                    marker.write_text("crashed")
                print("mock: simulated crash", flush=True)
                os._exit(3)

    async def completions(req: web.Request) -> web.StreamResponse:
        body = await req.json()
        prompt = body.get("prompt", "")
        max_tokens = int(body.get("max_tokens", 16))
        ignore_eos = bool(body.get("ignore_eos", False)) and not a.no_ignore_eos
        n_out = max_tokens if ignore_eos or not a.eos_at else min(max_tokens, a.eos_at)
        n_in = len(prompt.split())
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        if a.close_after_stream:
            resp.force_close()
        await resp.prepare(req)
        async with slots:
            state["served"] += 1
            state["active"] += 1
            try:
                await asyncio.sleep((a.ttft_ms + a.prefill_us_per_token * n_in / 1000.0) / 1000.0)
                t = time.monotonic()
                for i in range(n_out):
                    maybe_crash()
                    chunk = {"id": "cmpl-mock", "object": "text_completion", "model": a.model,
                             "choices": [{"index": 0, "text": f" t{i}",
                                          "finish_reason": "length" if i == n_out - 1 else None}]}
                    await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                    if a.spin_ms:
                        end = time.perf_counter() + a.spin_ms / 1000.0
                        while time.perf_counter() < end:
                            pass
                    tpot = a.tpot_ms * (1 + a.tpot_scale * (state["active"] - 1)) / 1000.0
                    t += tpot
                    await asyncio.sleep(max(0.0, t - time.monotonic()))
            finally:
                state["active"] -= 1
        if include_usage:
            usage = {"id": "cmpl-mock", "choices": [],
                     "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                               "total_tokens": n_in + n_out}}
            await resp.write(b"data: " + json.dumps(usage).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/completions", completions)
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--ttft-ms", type=float, default=20.0)
    ap.add_argument("--prefill-us-per-token", type=float, default=50.0)
    ap.add_argument("--tpot-ms", type=float, default=10.0)
    ap.add_argument("--tpot-scale", type=float, default=0.02)
    ap.add_argument("--spin-ms", type=float, default=0.0, help="busy-loop per token (CPU load)")
    ap.add_argument("--startup-delay-s", type=float, default=0.5)
    ap.add_argument("--crash-after", type=int, default=0, help="exit after serving N requests")
    ap.add_argument("--crash-once-file", default=None, help="only crash if this marker file is absent")
    ap.add_argument("--no-ignore-eos", action="store_true", help="simulate an engine ignoring ignore_eos")
    ap.add_argument("--eos-at", type=int, default=0)
    ap.add_argument("--close-after-stream", action="store_true",
                    help="close the connection after each streamed response (like llama-server)")
    ap.add_argument("--backend", choices=["zendnn", "cpu"], default="zendnn",
                    help="backend name printed at startup (the harness checks it against accel)")
    ap.add_argument("--accel-speedup", type=float, default=1.0,
                    help="with --backend zendnn, divide TTFT/TPOT by this factor")
    a = ap.parse_args()
    if a.backend == "zendnn" and a.accel_speedup != 1.0:
        a.ttft_ms /= a.accel_speedup
        a.prefill_us_per_token /= a.accel_speedup
        a.tpot_ms /= a.accel_speedup
    print(f"mock backend starting on {a.host}:{a.port} (parallel={a.parallel}, backend={a.backend})", flush=True)
    web.run_app(build_app(a), host=a.host, port=a.port, print=None, access_log=None)


if __name__ == "__main__":
    sys.exit(main())
