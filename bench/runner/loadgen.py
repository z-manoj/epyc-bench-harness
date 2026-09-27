"""Async OpenAI-compatible streaming load generator (shared by all engines)."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import aiohttp

from bench.core import status as S
from bench.runner.dataset import Prompt


@dataclass
class Endpoint:
    base_url: str          # e.g. http://127.0.0.1:8000
    model: str
    instance: int


@dataclass
class LoadResult:
    records: list[dict[str, Any]]
    abort_reason: Optional[str] = None
    abort_detail: str = ""
    t_start_ns: int = 0
    t_end_ns: int = 0
    wall_anchor: dict[str, float] = field(default_factory=dict)

    @property
    def ok_records(self) -> list[dict[str, Any]]:
        return [r for r in self.records if not r["error"]]


def _new_record(req_id: str, phase: str, ep: Endpoint, prompt: Prompt, output_len: int,
                t_submit: int) -> dict[str, Any]:
    return {
        "request_id": req_id, "phase": phase, "instance": ep.instance, "prompt_id": prompt.id,
        "nominal_prompt_tokens": prompt.nominal_tokens, "output_len": output_len,
        "t_submit": t_submit, "t_send": 0, "t_first_token": 0, "t_done": 0, "chunk_ts": [],
        "n_chunks": 0, "prompt_tokens": None, "completion_tokens": None,
        "completion_tokens_source": None, "http_status": None, "error": "",
    }


def _usage_from_chunk(obj: dict[str, Any], rec: dict[str, Any]) -> None:
    usage = obj.get("usage")
    if isinstance(usage, dict):
        if usage.get("prompt_tokens") is not None:
            rec["prompt_tokens"] = int(usage["prompt_tokens"])
        if usage.get("completion_tokens") is not None:
            rec["completion_tokens"] = int(usage["completion_tokens"])
            rec["completion_tokens_source"] = "usage"
    timings = obj.get("timings")   # llama.cpp
    if isinstance(timings, dict):
        if rec["prompt_tokens"] is None and timings.get("prompt_n") is not None:
            # prompt_n excludes cached tokens; only trust it when no cache was used
            rec["prompt_tokens"] = int(timings["prompt_n"]) + int(obj.get("tokens_cached", 0) or 0)
        if rec["completion_tokens"] is None and timings.get("predicted_n") is not None:
            rec["completion_tokens"] = int(timings["predicted_n"])
            rec["completion_tokens_source"] = "timings"
    if rec["prompt_tokens"] is None and obj.get("tokens_evaluated") is not None:
        rec["prompt_tokens"] = int(obj["tokens_evaluated"])


async def send_request(session: aiohttp.ClientSession, ep: Endpoint, prompt: Prompt, output_len: int,
                       rec: dict[str, Any], timeout_s: float, extra_body: dict[str, Any]) -> None:
    body = {
        "model": ep.model, "prompt": prompt.text, "max_tokens": output_len, "temperature": 0,
        "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True},
        **extra_body,
    }
    rec["t_send"] = time.monotonic_ns()
    chunk_ts: list[int] = rec["chunk_ts"]
    try:
        async with session.post(f"{ep.base_url}/v1/completions", json=body,
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            rec["http_status"] = resp.status
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:500]}"
                rec["t_done"] = time.monotonic_ns()
                return
            async for raw in resp.content:
                now = time.monotonic_ns()
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                if "error" in obj and not obj.get("choices"):
                    rec["error"] = f"stream error: {str(obj['error'])[:500]}"
                    break
                choices = obj.get("choices") or []
                if choices and (choices[0].get("text") or ""):
                    if not rec["t_first_token"]:
                        rec["t_first_token"] = now
                    chunk_ts.append(now)
                _usage_from_chunk(obj, rec)
            rec["t_done"] = time.monotonic_ns()
    except asyncio.TimeoutError:
        rec["error"] = "timeout"
        rec["t_done"] = time.monotonic_ns()
    except asyncio.CancelledError:
        rec["error"] = rec["error"] or "aborted"
        rec["t_done"] = time.monotonic_ns()
        raise
    except aiohttp.ClientError as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:500]
        rec["t_done"] = time.monotonic_ns()
    rec["n_chunks"] = len(chunk_ts)
    if not rec["error"]:
        if rec["completion_tokens"] is None:
            rec["completion_tokens"] = len(chunk_ts)
            rec["completion_tokens_source"] = "chunks"
        if not rec["t_first_token"]:
            rec["error"] = "no tokens streamed"


async def run_load(endpoints: list[Endpoint], prompts: list[Prompt], output_len: int, *,
                   concurrency: int, num_requests: int, min_duration_s: float = 0.0,
                   mode: str = "closed", phase: str = "measure", id_prefix: str = "r",
                   timeout_s: float = 1800.0, max_error_frac: float = 0.01,
                   abort_check: Optional[Callable[[], Optional[str]]] = None,
                   extra_body: Optional[dict[str, Any]] = None) -> LoadResult:
    """Run a closed-loop (online) or batch workload.

    closed: keep exactly `concurrency` requests in flight until `num_requests` have been issued
            and `min_duration_s` has elapsed.
    batch:  all `num_requests` are submitted at t0; at most `concurrency` (total slots) in flight.
    Requests are distributed round-robin across endpoints.
    """
    if not endpoints or not prompts:
        raise ValueError("need at least one endpoint and one prompt")
    extra_body = extra_body or {}
    records: list[dict[str, Any]] = []
    result = LoadResult(records, wall_anchor={"wall": time.time(), "mono_ns": time.monotonic_ns()})
    next_idx = 0
    errors = 0
    error_budget = max_error_frac * num_requests
    t0 = time.monotonic_ns()
    result.t_start_ns = t0
    min_ns = int(min_duration_s * 1e9) if mode == "closed" else 0
    stop = asyncio.Event()

    def abort(reason: str, detail: str) -> None:
        if result.abort_reason is None:
            result.abort_reason, result.abort_detail = reason, detail
        stop.set()

    async def worker(session: aiohttp.ClientSession) -> None:
        nonlocal next_idx, errors
        while not stop.is_set():
            i = next_idx
            if i >= num_requests and (time.monotonic_ns() - t0) >= min_ns:
                return
            next_idx += 1
            ep = endpoints[i % len(endpoints)]
            prompt = prompts[i % len(prompts)]
            rec = _new_record(f"{id_prefix}{i:06d}", phase, ep, prompt, output_len,
                              t0 if mode == "batch" else time.monotonic_ns())
            records.append(rec)
            await send_request(session, ep, prompt, output_len, rec, timeout_s, extra_body)
            if rec["error"]:
                errors += 1
                if rec["error"] == "timeout":
                    abort(S.FAILED_TIMEOUT, f"request {rec['request_id']} exceeded {timeout_s}s")
                elif errors > error_budget:
                    abort(S.FAILED_ERRORS, f"{errors} errors (last: {rec['error']})")

    async def watchdog() -> None:
        while not stop.is_set():
            reason = abort_check() if abort_check else None
            if reason:
                abort(reason, "abort_check triggered")
                return
            try:
                await asyncio.wait_for(stop.wait(), 0.5)
            except asyncio.TimeoutError:
                pass

    # One connection per request: llama-server closes the connection after a streamed response, so a
    # pooled keep-alive connection fails on reuse. Same behaviour for every engine keeps TTFT comparable.
    connector = aiohttp.TCPConnector(limit=0, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [asyncio.create_task(worker(session)) for _ in range(max(1, concurrency))]
        wd = asyncio.create_task(watchdog())
        stopper = asyncio.create_task(stop.wait())
        try:
            pending = set(workers)
            while pending:
                done, _ = await asyncio.wait(pending | {stopper}, return_when=asyncio.FIRST_COMPLETED)
                pending -= done
                if stopper in done:
                    break
            if result.abort_reason:
                for w in workers:
                    w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        finally:
            stop.set()
            for t in (wd, stopper, *workers):
                t.cancel()
            await asyncio.gather(wd, stopper, return_exceptions=True)
    result.t_end_ns = time.monotonic_ns()
    for r in records:
        r["n_chunks"] = len(r["chunk_ts"])
    return result
