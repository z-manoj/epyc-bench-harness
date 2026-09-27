"""The run state machine (spec Section 5): server launch, stability gate, warmup, measurement,
cooldown and intra-rep validation, plus the per-point repetition loop (Section 7)."""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from bench import __version__
from bench.analysis import consistency, gates
from bench.analysis import metrics as rm
from bench.core import status as S
from bench.core.config import ServerConfig, WorkloadPoint, expand_points
from bench.core.io import EventLog, _json_default, log, write_json
from bench.runner.loadgen import LoadResult, run_load
from bench.runner.server import BACKEND_ENFORCED, ServerGroup, ServerStartError, _pdeathsig
from bench.runner.storage import clean_incomplete_reps, rep_history, write_requests
from bench.system.env import diff_tunables, tunables
from bench.system.samples import SampleStore, finalize_samples

if TYPE_CHECKING:
    from bench.runner.experiment import Experiment


def read_oom_kills() -> int:
    try:
        for line in Path("/proc/vmstat").read_text().splitlines():
            if line.startswith("oom_kill "):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


@dataclass
class RepOutcome:
    status: str
    flags: list[str] = field(default_factory=list)
    server_dead: bool = False


class ConfigAborted(Exception):
    def __init__(self, status: str, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


class ServerConfigRunner:
    def __init__(self, exp: "Experiment", sc: ServerConfig) -> None:
        self.exp = exp
        self.cfg = exp.cfg
        self.sc = sc
        self.dir = exp.dir / sc.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events = EventLog(self.dir / "events.jsonl")
        self.points = expand_points(self.cfg, sc)
        self.server: Optional[ServerGroup] = None
        self.sampler_proc: Optional[subprocess.Popen] = None
        self.store: Optional[SampleStore] = None
        self.ctl_path = self.dir / "samples" / "sampler_ctl.json"
        self.launch_count = 0
        harness = set(self.cfg.platform.harness_cores)
        self.harness = sorted(harness)
        self.server_cores = sc.all_cores
        self.instance_cores = [i.core_list for i in sc.instances]
        self.cpu_ids: list[int] = sorted(os.sched_getaffinity(0) | set(range(os.cpu_count() or 1)))
        self.other_cores = [c for c in self.cpu_ids if c not in harness and c not in set(self.server_cores)]
        ref = self.dir / "gate_reference.json"
        self.ref_freq: Optional[float] = json.loads(ref.read_text()).get("freq_mean_mhz") if ref.exists() else None

    # ------------------------------------------------------------------ helpers
    def point_dir(self, p: WorkloadPoint) -> Path:
        return self.dir / p.point_id

    def decide(self, p: WorkloadPoint) -> dict[str, Any]:
        return consistency.decide_point(rep_history(self.point_dir(p)), self.cfg.run_policy,
                                  self.cfg.thresholds.consistency)

    def write_status(self, status: str, detail: str = "") -> None:
        write_json(self.dir / "config_status.json",
                   {"status": status, "detail": detail, "wall": time.time(), "launches": self.launch_count})

    def write_point_summary(self, p: WorkloadPoint) -> dict[str, Any]:
        hist = rep_history(self.point_dir(p))
        d = consistency.decide_point(hist, self.cfg.run_policy, self.cfg.thresholds.consistency)
        summary = {
            "server_config_id": self.sc.id, **p.as_dict(),
            "classification": d.get("classification", S.NOT_RUN) if d["action"] == "done" else S.NOT_RUN,
            "decision": d,
            "reps": [{"rep": r["rep"], "status": r["status"], "flags": r.get("flags", []),
                      "status_reasons": r.get("status_reasons", [])} for r in hist],
        }
        self.point_dir(p).mkdir(parents=True, exist_ok=True)
        write_json(self.point_dir(p) / "point_summary.json", summary)
        return summary

    # ------------------------------------------------------------------ sampler
    def _write_ctl(self, roots: list[int], stop: bool = False) -> None:
        tmp = self.ctl_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"roots": roots, "stop": stop}))
        os.replace(tmp, self.ctl_path)

    async def start_sampler(self) -> None:
        sdir = self.dir / "samples"
        sdir.mkdir(exist_ok=True)
        self._write_ctl([])
        out = sdir / f"session_{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}.bin"
        cmd = [sys.executable, "-m", "bench.system.sampler", "--out", str(out),
               "--interval-ms", str(self.cfg.run_policy.sample_interval_ms),
               "--ctl", str(self.ctl_path), "--cores", ",".join(map(str, self.harness))]
        self.sampler_proc = subprocess.Popen(cmd, start_new_session=True, preexec_fn=_pdeathsig,
                                             stderr=open(sdir / "sampler.log", "ab"),
                                             env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)})
        self.store = SampleStore(out)
        if not await self.store.wait_for(time.monotonic_ns(), timeout_s=30):
            raise ConfigAborted(S.INVALID_SYSTEM, "sampler did not produce samples")
        self.events.write("sampler", "start", pid=self.sampler_proc.pid, file=out.name)

    def stop_sampler(self) -> None:
        if self.sampler_proc is None:
            return
        try:
            self._write_ctl([], stop=True)
            self.sampler_proc.send_signal(signal.SIGTERM)
            self.sampler_proc.wait(timeout=10)
        except (subprocess.TimeoutExpired, OSError):
            self.sampler_proc.kill()
            self.sampler_proc.wait()
        self.events.write("sampler", "stop")
        self.sampler_proc = None
        finalize_samples(self.dir)

    async def samples(self, t0: int, t1: int):
        assert self.store is not None
        await self.store.wait_for(t1, timeout_s=10)
        return self.store.window(t0, t1)

    # ------------------------------------------------------------------ server
    async def system_check(self) -> None:
        """Pre-launch: machine idle with no server running (5.0.1)."""
        window = self.cfg.run_policy.system_check_window_s or self.cfg.run_policy.stability_window_s
        thr = self.cfg.thresholds.system_check
        own = {os.getpid()} | {c.pid for c in psutil.Process().children(recursive=True)}

        def snap() -> dict[int, tuple[str, float]]:
            out = {}
            for pr in psutil.process_iter(["pid", "name", "cpu_times"]):
                try:
                    ct = pr.info["cpu_times"]
                    if ct is not None:
                        out[pr.info["pid"]] = (pr.info["name"], ct.user + ct.system)
                except (psutil.Error, TypeError):
                    pass
            return out

        t0 = self.events.write("system_check", "start")
        a = snap()
        await asyncio.sleep(window)
        b = snap()
        t1 = self.events.write("system_check", "end")
        procs = sorted(((pid, name, (cpu - a[pid][1]) / window * 100.0)
                        for pid, (name, cpu) in b.items() if pid in a),
                       key=lambda x: -x[2])
        foreign = [(pid, n, c) for pid, n, c in procs if pid not in own and c > thr.max_foreign_proc_cpu_pct]
        s = await self.samples(t0, t1)
        non_harness = [c for c in self.cpu_ids if c not in set(self.harness)]
        util = gates._nanmean(s.util(non_harness))
        result = {
            "window_s": window, "non_harness_util_mean": util,
            "missed_frac": s.missed_frac(),
            "top_processes": [{"pid": p, "name": n, "cpu_pct": round(c, 2)} for p, n, c in procs[:15]],
            "foreign_over_threshold": [{"pid": p, "name": n, "cpu_pct": round(c, 2)} for p, n, c in foreign],
            "thresholds": thr.model_dump(),
        }
        reasons = []
        if util is None or util > thr.max_mean_util_pct:
            reasons.append(f"non-harness core util {util} > {thr.max_mean_util_pct}%")
        if foreign:
            reasons.append("foreign processes: " + ", ".join(f"{n}[{p}] {c:.1f}%" for p, n, c in foreign[:5]))
        result["passed"] = not reasons
        result["reasons"] = reasons
        with open(self.dir / "system_check.jsonl", "a") as f:
            f.write(json.dumps({"wall": time.time(), **result}, default=_json_default) + "\n")
        if reasons:
            log(f"[{self.sc.id}] system check FAILED: {'; '.join(reasons)}")
            raise ConfigAborted(S.INVALID_SYSTEM, "; ".join(reasons))

    async def launch(self) -> None:
        self.server = ServerGroup(self.sc, self.dir, self.cfg.platform.require_numactl)
        self.launch_count += 1
        t0 = time.monotonic()
        if self.sc.mode == "attach":
            urls = [f"http://{e.host}:{e.port}" for e in self.sc.endpoints]
            self.events.write("server", "attach", launch=self.launch_count, urls=urls)
            log(f"[{self.sc.id}] attaching to {', '.join(urls)}")
            try:
                self.server.attach()
                await self.server.wait_ready(self.cfg.run_policy.server_ready_timeout_s)
            except ServerStartError as e:
                self.server.write_launch({"error": str(e)})
                raise ConfigAborted(S.FAILED_SERVER_START, str(e)) from e
        else:
            await self.system_check()
            for i in range(len(self.sc.instances)):
                if not port_free(self.sc.host, self.sc.port(i)):
                    raise ConfigAborted(S.FAILED_SERVER_START, f"port {self.sc.port(i)} already in use")
            self.events.write("server", "launch", launch=self.launch_count)
            log(f"[{self.sc.id}] launching {len(self.sc.instances)} instance(s)")
            try:
                self.server.start()
                self._write_ctl(self.server.pids())
                await self.server.wait_ready(self.cfg.run_policy.server_ready_timeout_s)
            except ServerStartError as e:
                self.server.write_launch({"error": str(e)})
                self.server.stop()
                raise ConfigAborted(S.FAILED_SERVER_START, str(e)) from e
            mismatch = self.server.backend_mismatch()
            if mismatch:
                if self.sc.engine in BACKEND_ENFORCED:
                    self.server.stop()
                    raise ConfigAborted(S.INVALID_BACKEND, mismatch)
                log(f"[{self.sc.id}] warning: {mismatch}")
        self.events.write("server", "ready", ready_s=time.monotonic() - t0)
        self.server.write_launch({"ready_total_s": time.monotonic() - t0, "launch": self.launch_count})
        for w in self.server.warnings:
            log(f"[{self.sc.id}] warning: {w}")
        await asyncio.sleep(1.5)   # let the sampler pick up a new process tree (launch mode)
        await self.server_warmup()

    async def server_warmup(self) -> None:
        """One-time warmup covering every prompt size at concurrency 1 (5.0.5)."""
        n = self.cfg.warmup.server_warmup_requests
        if n <= 0 or self.server is None:
            return
        sizes = sorted({p.prompt_size for p in self.points}, key=lambda s: (isinstance(s, str), s))
        out_len = min(p.output_len for p in self.points)
        per = max(1, math.ceil(n / len(sizes)))
        self.events.write("server_warmup", "start", launch=self.launch_count)
        records = []
        for size in sizes:
            _, warm_pool = self.exp.dataset.split(size)
            res = await run_load(self.server.endpoints(), warm_pool, out_len, concurrency=1, num_requests=per,
                                 phase="server_warmup", id_prefix=f"sw{size}_",
                                 timeout_s=self.cfg.run_policy.request_timeout_s, max_error_frac=1.0,
                                 abort_check=self.abort_check, extra_body=self.server.extra_body())
            for r in res.records:
                r["prompt_size"] = str(size)
            records += res.records
            if res.abort_reason in (S.FAILED_CRASH, S.FAILED_OOM):
                break
        self.events.write("server_warmup", "end", launch=self.launch_count)
        path = self.dir / "server_warmup.parquet"
        if records:
            tmp = self.dir / "server_warmup.new.parquet"
            write_requests(tmp, records, {"launch": self.launch_count, "prompt_size": ""})
            if path.exists():
                merged = pa.concat_tables([pq.read_table(path), pq.read_table(tmp)], promote_options="default")
                pq.write_table(merged, path)
                tmp.unlink()
            else:
                os.replace(tmp, path)
        ok = [r for r in records if not r["error"]]
        log(f"[{self.sc.id}] server warmup: {len(ok)}/{len(records)} ok")
        if self.server.dead():
            raise ConfigAborted(S.FAILED_SERVER_START, "server died during one-time warmup")

    def stop_server(self) -> None:
        if self.server is not None:
            external = self.server.external
            self.server.stop()
            self.events.write("server", "detach" if external else "stop")
            self._write_ctl([])
            self.server = None

    async def relaunch(self) -> None:
        self.stop_server()
        await self.launch()

    def abort_check(self) -> Optional[str]:
        if self.server is None or self.server.dead():
            return S.FAILED_CRASH
        if read_oom_kills() > self._oom0:
            return S.FAILED_OOM
        return None

    _oom0 = 0

    # ------------------------------------------------------------------ main
    async def run(self) -> None:
        todo = []
        for p in self.points:
            removed = clean_incomplete_reps(self.point_dir(p))
            if removed:
                log(f"[{self.sc.id}] {p.point_id}: removed incomplete rep(s) {removed}")
            if self.decide(p)["action"] == "run":
                todo.append(p)
            else:
                self.write_point_summary(p)
        if not todo:
            log(f"[{self.sc.id}] all {len(self.points)} points complete; skipping")
            self.write_status(S.CONFIG_DONE)
            return

        changed = diff_tunables(self.exp.baseline_tunables, tunables())
        if changed:
            self.write_status(S.INVALID_ENV_CHANGED, json.dumps(changed))
            log(f"[{self.sc.id}] ABORT {S.INVALID_ENV_CHANGED}: {changed}")
            return

        log(f"[{self.sc.id}] {len(todo)}/{len(self.points)} points to run")
        try:
            await self.start_sampler()
            await self.launch()
            for p in todo:
                await self.run_point(p)
            self.write_status(S.CONFIG_DONE)
        except ConfigAborted as e:
            log(f"[{self.sc.id}] ABORT {e.status}: {e.detail}")
            self.write_status(e.status, e.detail)
            for p in self.points:
                if (self.point_dir(p)).exists():
                    self.write_point_summary(p)
        finally:
            self.stop_server()
            self.stop_sampler()

    async def run_point(self, p: WorkloadPoint) -> None:
        pdir = self.point_dir(p)
        pdir.mkdir(parents=True, exist_ok=True)
        while True:
            d = self.decide(p)
            if d["action"] != "run":
                s = self.write_point_summary(p)
                log(f"[{self.sc.id}] {p.point_id}: {s['classification']}")
                return
            hist = rep_history(pdir)
            rep = (max((h["rep"] for h in hist), default=0)) + 1
            outcome = await self.run_rep(p, rep)
            log(f"[{self.sc.id}] {p.point_id} rep {rep}: {outcome.status} {' '.join(outcome.flags)}")
            if outcome.server_dead or (self.server is not None and self.server.dead()):
                if self.sc.mode == "attach":
                    raise ConfigAborted(S.FAILED_CRASH, "attached server stopped responding; it is not restarted")
                log(f"[{self.sc.id}] server died; relaunching")
                await self.relaunch()
            elif self.cfg.run_policy.restart_server_between_reps and self.sc.mode == "launch":
                await self.relaunch()

    # ------------------------------------------------------------------ one rep
    async def stability_gate(self, pid: str, rep: int) -> dict[str, Any]:
        rp, thr = self.cfg.run_policy, self.cfg.thresholds.gate
        attempts = []
        passed = False
        for a in range(1 + rp.stability_max_retries):
            if self.abort_check() == S.FAILED_CRASH:
                return {"passed": False, "crashed": True, "attempts": attempts}
            t0 = self.events.write("gate", "start", pid, rep, attempt=a)
            await asyncio.sleep(rp.stability_window_s)
            t1 = self.events.write("gate", "end", pid, rep, attempt=a)
            st = gates.gate_stats(await self.samples(t0, t1), self.server_cores, self.other_cores)
            ok, reasons = gates.evaluate_gate(st, thr, self.ref_freq, attribute_cores=bool(self.server_cores))
            attempts.append({"attempt": a, "t0_ns": t0, "t1_ns": t1, "stats": st, "passed": ok,
                             "reasons": reasons})
            if ok:
                passed = True
                if self.ref_freq is None and st.get("freq_mean_mhz"):
                    self.ref_freq = st["freq_mean_mhz"]
                    write_json(self.dir / "gate_reference.json",
                               {"freq_mean_mhz": self.ref_freq, "point_id": pid, "rep": rep})
                break
            log(f"[{self.sc.id}] {pid} rep {rep}: gate attempt {a} failed: {'; '.join(reasons)}")
        return {"passed": passed, "attempts": attempts, "baseline": attempts[-1]["stats"] if attempts else {},
                "ref_freq_mhz": self.ref_freq, "thresholds": thr.model_dump()}

    async def run_rep(self, p: WorkloadPoint, rep: int) -> RepOutcome:
        cfg, rp, vt = self.cfg, self.cfg.run_policy, self.cfg.thresholds.validation
        pid = p.point_id
        rdir = self.point_dir(p) / f"rep_{rep}"
        rdir.mkdir(parents=True, exist_ok=True)
        assert self.server is not None and self.store is not None
        t_rep = self.events.write("rep", "start", pid, rep)
        self.store.trim(t_rep - int(5e9))
        flags: list[str] = []
        reasons: list[str] = []
        phases: dict[str, Any] = {"rep_start": t_rep}
        summary: dict[str, Any] = {
            "server_config_id": self.sc.id, "engine": self.sc.engine, "precision": self.sc.precision,
            "topology": self.sc.topology, "accel": self.sc.accel, "n_server_cores": len(self.server_cores), **p.as_dict(),
            "rep": rep, "harness_version": __version__, "phases": phases,
        }

        def finish(status: str, dead: bool = False) -> RepOutcome:
            phases["rep_end"] = self.events.write("rep", "end", pid, rep, status=status)
            summary.update(status=status, flags=sorted(set(flags)), status_reasons=reasons)
            write_json(rdir / "summary.json", summary)
            return RepOutcome(status, sorted(set(flags)), dead)

        # 1. stability gate
        gate = await self.stability_gate(pid, rep)
        write_json(rdir / "gate.json", gate)
        summary["gate"] = {"passed": gate["passed"], "attempts": len(gate["attempts"]),
                           "baseline": gate.get("baseline")}
        if gate.get("crashed"):
            reasons.append("server not running before gate")
            return finish(S.FAILED_CRASH, dead=True)
        if not gate["passed"]:
            reasons.append("stability gate failed: " + "; ".join(gate["attempts"][-1]["reasons"]))
            return finish(S.INVALID_BASELINE)
        baseline = gate["baseline"]

        # 2. warmup (same shape, disjoint prompts)
        meas_pool, warm_pool = self.exp.dataset.split(p.prompt_size)
        eps = self.server.endpoints()
        extra = self.server.extra_body()
        self._oom0 = read_oom_kills()
        phases["warmup_start"] = self.events.write("warmup", "start", pid, rep)
        warm = await run_load(eps, warm_pool, p.output_len, concurrency=p.concurrency,
                              num_requests=cfg.warmup.per_rep_warmup_requests,
                              min_duration_s=cfg.warmup.per_rep_warmup_min_s, mode="closed", phase="warmup",
                              id_prefix="w", timeout_s=rp.request_timeout_s, max_error_frac=vt.max_error_frac,
                              abort_check=self.abort_check, extra_body=extra)
        phases["warmup_end"] = self.events.write("warmup", "end", pid, rep, n=len(warm.records))

        meas: Optional[LoadResult] = None
        failure: Optional[str] = None
        if warm.abort_reason:
            failure = warm.abort_reason
            reasons.append(f"warmup aborted: {warm.abort_reason} {warm.abort_detail}")
        else:
            # 3. measurement
            phases["measure_start"] = self.events.write("measure", "start", pid, rep)
            meas = await run_load(eps, meas_pool, p.output_len, concurrency=p.concurrency,
                                  num_requests=p.num_requests,
                                  min_duration_s=p.min_measure_s if p.scenario != "batch" else 0.0,
                                  mode="batch" if p.scenario == "batch" else "closed", phase="measure",
                                  id_prefix="m", timeout_s=rp.request_timeout_s, max_error_frac=vt.max_error_frac,
                                  abort_check=self.abort_check, extra_body=extra)
            phases["measure_end"] = self.events.write("measure", "end", pid, rep, n=len(meas.records),
                                                      abort=meas.abort_reason)
            summary["wall_anchor"] = meas.wall_anchor
            if meas.abort_reason:
                failure = meas.abort_reason
                reasons.append(f"measurement aborted: {meas.abort_reason} {meas.abort_detail}")

        if failure:
            await asyncio.sleep(1.0)   # connection resets can arrive before the dead process is reaped
        dead = bool(self.server.dead())
        if read_oom_kills() > self._oom0:
            failure = S.FAILED_OOM
            reasons.append("kernel oom_kill counter increased")
        elif dead:
            failure = S.FAILED_CRASH
            reasons.append(f"server instance(s) {self.server.dead()} exited")

        records = list(warm.records) + (list(meas.records) if meas else [])
        write_requests(rdir / "requests.parquet", records, {"point_id": pid, "rep": rep})

        # 4. cooldown
        phases["cooldown_start"] = self.events.write("cooldown", "start", pid, rep)
        cool = {}
        if not dead:
            cool = await self.cooldown(phases["cooldown_start"], baseline, flags)
        phases["cooldown_end"] = self.events.write("cooldown", "end", pid, rep)
        summary["cooldown"] = cool

        if meas is None or not meas.records:
            return finish(failure or S.FAILED_ERRORS, dead)

        # 5. validation
        mrecs = meas.records
        metrics = rm.request_metrics(mrecs, p.scenario, p.concurrency, len(self.server_cores))
        t0m, t1m = metrics["window_start_ns"], metrics["window_end_ns"]
        s_meas = await self.samples(t0m, t1m)
        cpu = gates.measurement_cpu_stats(s_meas, self.server_cores, self.instance_cores, self.other_cores,
                                          baseline.get("rss_mean_mb"))
        metrics.update(cpu)
        metrics["recovery_s"] = cool.get("recovery_s")
        metrics["rss_growth_mb"] = cool.get("rss_growth_mb")
        summary["metrics"] = metrics

        checks: dict[str, Any] = {}
        if failure is None:
            if metrics["error_frac"] and metrics["error_frac"] > vt.max_error_frac:
                failure = S.FAILED_ERRORS
                reasons.append(f"error fraction {metrics['error_frac']:.3f} > {vt.max_error_frac}")
            elif any(r["error"] == "timeout" for r in mrecs):
                failure = S.FAILED_TIMEOUT
            if (cpu.get("sys_mem_available_min_mb") is not None and s_meas.n
                    and cpu["sys_mem_available_min_mb"] < vt.oom_min_available_frac * float(s_meas.mem["mem_total_mb"][0])):
                failure = S.FAILED_OOM
                reasons.append("system MemAvailable dropped below the OOM threshold")

        tok = rm.token_checks(mrecs, p.output_len, vt)
        checks["tokens"] = tok
        if tok["prompt_warn"]:
            flags.append(S.WARN_PROMPT_TOKENS)
        if tok["prompt_unknown"] and tok["prompt_unknown"] >= metrics["n_ok"]:
            flags.append(S.WARN_NO_PROMPT_USAGE)

        s0, s1 = metrics["steady_start_ns"], metrics["steady_end_ns"]
        drift = rm.drift_check(mrecs, s0, s1, vt.max_drift_frac, gate_tpot=p.scenario != "batch")
        checks["drift"] = drift
        outside = cpu.get("cpu_outside_util_mean") or 0.0
        checks["pinning"] = {"outside_util_mean": outside, "limit": vt.max_outside_util_pct,
                             "passed": outside <= vt.max_outside_util_pct}
        f_meas, f_gate = cpu.get("freq_mean_mhz"), baseline.get("freq_mean_mhz")
        throttle = bool(f_meas and f_gate and f_meas < f_gate * (1 - vt.max_freq_drop_frac))
        checks["frequency"] = {"measure_mhz": f_meas, "gate_mhz": f_gate, "throttle": throttle}
        if throttle:
            flags.append(S.WARN_THROTTLE)
        checks["sampler"] = {"missed_frac": cpu["missed_frac"], "limit": vt.max_missed_frac,
                             "passed": cpu["missed_frac"] is not None and cpu["missed_frac"] <= vt.max_missed_frac}
        if p.min_measure_s and metrics["window_s"] < 0.9 * p.min_measure_s:
            flags.append(S.WARN_SHORT_WINDOW)
        summary["checks"] = checks

        if failure:
            status = failure
        elif tok["prompt_fail"]:
            status = S.FAILED_INPUT
            reasons.append(f"prompt token deviation {tok['prompt_dev_max']:.3f} > {vt.prompt_fail_frac}")
        elif tok["tokens_fail"]:
            status = S.FAILED_TOKENS
            reasons.append(f"only {tok['complete_frac']:.3f} of requests produced exactly {p.output_len} tokens "
                           f"(engine may not honour ignore_eos)")
        elif not checks["sampler"]["passed"]:
            status = S.INVALID_SAMPLER
            reasons.append(f"sampler missed {cpu['missed_frac']:.3%} of samples")
        elif self.server_cores and not checks["pinning"]["passed"]:
            status = S.INVALID_PINNING
            reasons.append(f"mean util outside pinned cores {outside:.2f}% > {vt.max_outside_util_pct}%")
        elif not drift["passed"]:
            status = S.UNSTABLE_DRIFT
            reasons.append(f"intra-rep drift tok/s {drift.get('tok_s_drift')} tpot {drift.get('tpot_p50_drift')}")
        else:
            status = S.SUCCESS
        return finish(status, dead)

    async def cooldown(self, t_start: int, baseline: dict[str, Any], flags: list[str]) -> dict[str, Any]:
        rp, vt = self.cfg.run_policy, self.cfg.thresholds.validation
        target = (baseline.get("server_util_mean") or 0.0) + vt.recovery_margin_pp
        await asyncio.sleep(rp.cooldown_s)
        waited = rp.cooldown_s
        smooth = max(1, int(1000 / rp.sample_interval_ms))
        while True:
            now = time.monotonic_ns()
            s = await self.samples(t_start, now)
            rec = gates.recovery_time(s, self.server_cores, t_start, target, smooth)
            if rec is not None or waited >= rp.cooldown_s * vt.max_cooldown_factor:
                break
            await asyncio.sleep(1.0)
            waited += 1.0
        if rec is None:
            flags.append(S.WARN_SLOW_RECOVERY)
        tail = s.slice(now - int(2e9), now)
        rss_end = gates._nanmean(tail.rss_total())
        rss_base = baseline.get("rss_mean_mb")
        growth = (rss_end - rss_base) if (rss_end is not None and rss_base is not None) else None
        if growth is not None and growth > vt.mem_growth_warn_mb:
            flags.append(S.WARN_MEM_GROWTH)
        return {"target_util_pct": target, "recovery_s": rec, "waited_s": waited,
                "rss_end_mb": rss_end, "rss_baseline_mb": rss_base, "rss_growth_mb": growth}


# ---------------------------------------------------------------------------
