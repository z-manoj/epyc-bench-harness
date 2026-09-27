"""Launch or attach, health-check, and stop llama-server, vllm serve and mock server instances.

mode=launch execs the binary. mode=attach only connects to host:port; it never signals the process.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import aiohttp

from bench.core.config import ServerConfig
from bench.runner.loadgen import Endpoint

BACKEND_MARKERS = {
    "llamacpp": re.compile(r"zendnn", re.I),
    "vllm": re.compile(r"zentorch|zencpu|zendnn", re.I),
    "mock": re.compile(r"backend=zendnn", re.I),
}
# Engines whose detection is reliable enough to reject a run on mismatch. vLLM only logs zentorch
# at some verbosity levels, so a mismatch there is a warning.
BACKEND_ENFORCED = {"llamacpp", "mock"}
LAUNCH_HEADER = "==== launch "


class ServerStartError(RuntimeError):
    pass


def _pdeathsig() -> None:
    """Child: receive SIGTERM if the harness dies; own process group for clean teardown."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
    except OSError:
        pass


@dataclass
class InstanceProc:
    idx: int
    port: int
    cores: list[int]
    numa_node: int
    cmd: list[str]
    env_overrides: dict[str, str]
    log_path: Path
    host: str = "127.0.0.1"
    proc: Optional[subprocess.Popen] = None
    model_name: Optional[str] = None
    ready_s: Optional[float] = None


@dataclass
class ServerGroup:
    sc: ServerConfig
    out_dir: Path
    require_numactl: bool = True
    instances: list[InstanceProc] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pinning_method: str = ""
    external: bool = False

    # ---------------------------------------------------------------- commands
    def _engine_cmd(self, idx: int) -> list[str]:
        sc = self.sc
        port = str(sc.port(idx))
        if sc.engine == "llamacpp":
            return [sc.binary, "-m", sc.model, "--host", sc.host, "--port", port, *sc.args]
        if sc.engine == "vllm":
            exe = shlex.split(sc.binary) if sc.binary else ["vllm"]
            cmd = [*exe, "serve", sc.model, "--host", sc.host, "--port", port]
            if sc.served_model_name:
                cmd += ["--served-model-name", sc.served_model_name]
            return cmd + sc.args
        # mock
        return [sys.executable, sc.binary, "--host", sc.host, "--port", port, *sc.args]

    def build(self) -> None:
        sc = self.sc
        numactl = shutil.which("numactl")
        if not numactl and self.require_numactl:
            raise ServerStartError("numactl not found (set platform.require_numactl: false to fall back "
                                   "to sched_setaffinity without memory binding)")
        self.pinning_method = "numactl" if numactl else "sched_setaffinity"
        if not numactl:
            self.warnings.append("numactl missing: CPU pinning via sched_setaffinity, no --membind")
        if sc.engine == "vllm" and "--no-enable-prefix-caching" not in sc.args:
            self.warnings.append("vLLM prefix caching not disabled (--no-enable-prefix-caching); "
                                 "repeated prompts across reps may hit the cache")
        self.instances = []
        for i, inst in enumerate(sc.instances):
            cores = inst.core_list
            env = dict(sc.env)
            if sc.engine == "vllm":
                env["VLLM_CPU_OMP_THREADS_BIND"] = inst.cores
            if sc.engine == "llamacpp":
                t = sc.arg_value("-t", "--threads")
                if t is not None and int(t) != len(cores):
                    self.warnings.append(f"instance {i}: -t {t} != {len(cores)} pinned cores")
                if t is None:
                    self.warnings.append(f"instance {i}: -t not set; llama.cpp picks its own thread count")
            cmd = self._engine_cmd(i)
            if numactl:
                cmd = [numactl, f"--physcpubind={inst.cores}", f"--membind={inst.numa_node}", *cmd]
            self.instances.append(InstanceProc(i, sc.port(i), cores, inst.numa_node, cmd, env,
                                               self.out_dir / f"server_{i}.log", host=sc.host))

    def attach(self) -> None:
        """Point at servers that are already listening. Does not exec anything."""
        sc = self.sc
        self.external = True
        self.pinning_method = "external"
        if not sc.instances:
            self.warnings.append("no cores declared; CPU pinning and server-core gates are skipped")
        self.instances = []
        for i, ep in enumerate(sc.endpoints):
            cores = sc.instances[i].core_list if i < len(sc.instances) else []
            numa = sc.instances[i].numa_node if i < len(sc.instances) else 0
            log_path = self.out_dir / f"server_{i}.log"
            log_path.write_text(
                f"{LAUNCH_HEADER}{time.strftime('%Y-%m-%d %H:%M:%S')}: attach http://{ep.host}:{ep.port}\n")
            self.instances.append(InstanceProc(i, ep.port, cores, numa, [], {}, log_path, host=ep.host))

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.build()
        for ip in self.instances:
            env = {**os.environ, **ip.env_overrides}
            logf = open(ip.log_path, "ab")
            logf.write(f"\n{LAUNCH_HEADER}{time.strftime('%Y-%m-%d %H:%M:%S')}: {shlex.join(ip.cmd)}\n".encode())
            logf.flush()
            cores = ip.cores
            use_affinity = self.pinning_method != "numactl"

            def pre(cores=cores, use_affinity=use_affinity) -> None:
                _pdeathsig()
                if use_affinity:
                    os.sched_setaffinity(0, cores)

            try:
                ip.proc = subprocess.Popen(ip.cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                                           start_new_session=True, preexec_fn=pre)
            except OSError as e:
                raise ServerStartError(f"instance {ip.idx}: cannot exec {ip.cmd[0]}: {e}") from e
            finally:
                logf.close()

    async def wait_ready(self, timeout_s: float) -> None:
        t0 = time.monotonic()
        async with aiohttp.ClientSession() as session:
            for ip in self.instances:
                base = f"http://{ip.host}:{ip.port}"
                while True:
                    if not self.external and (ip.proc is None or ip.proc.poll() is not None):
                        raise ServerStartError(f"instance {ip.idx} exited during startup "
                                               f"(rc={ip.proc.returncode if ip.proc else None}); see {ip.log_path}")
                    if time.monotonic() - t0 > timeout_s:
                        raise ServerStartError(f"instance {ip.idx} not ready after {timeout_s}s")
                    if await self._probe(session, base, ip):
                        ip.ready_s = time.monotonic() - t0
                        break
                    await asyncio.sleep(1.0)

    async def _probe(self, session: aiohttp.ClientSession, base: str, ip: InstanceProc) -> bool:
        tmo = aiohttp.ClientTimeout(total=5)
        try:
            async with session.get(f"{base}/health", timeout=tmo) as r:
                if r.status != 200:
                    return False
            async with session.get(f"{base}/v1/models", timeout=tmo) as r:
                if r.status != 200:
                    return self.sc.engine == "llamacpp" and self._fallback_model(ip)
                data = await r.json(content_type=None)
                models = data.get("data") or []
                ip.model_name = self.sc.served_model_name or (models[0]["id"] if models else self.sc.model)
                return True
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError):
            return False

    def _fallback_model(self, ip: InstanceProc) -> bool:
        ip.model_name = self.sc.served_model_name or self.sc.model
        return True

    def pids(self) -> list[int]:
        return [ip.proc.pid for ip in self.instances if ip.proc is not None]

    def dead(self) -> list[int]:
        """Indices of instances whose process has exited. Attached servers are not owned."""
        if self.external:
            return []
        return [ip.idx for ip in self.instances if ip.proc is None or ip.proc.poll() is not None]

    def endpoints(self) -> list[Endpoint]:
        return [Endpoint(f"http://{ip.host}:{ip.port}", ip.model_name or self.sc.model, ip.idx)
                for ip in self.instances]

    def stop(self, grace_s: float = 30.0) -> None:
        if self.external:
            return
        for ip in self.instances:
            if ip.proc and ip.proc.poll() is None:
                try:
                    os.killpg(ip.proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + grace_s
        for ip in self.instances:
            if ip.proc is None:
                continue
            try:
                ip.proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(ip.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                ip.proc.wait()
            # Make sure the whole group (engine worker processes) is gone.
            try:
                os.killpg(ip.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    # ---------------------------------------------------------------- metadata
    def list_devices(self) -> str:
        """llama-server only names its backends at default verbosity via --list-devices."""
        if self.sc.engine != "llamacpp" or not self.sc.binary:
            return ""
        if self._devices is None:
            try:
                p = subprocess.run([self.sc.binary, "--list-devices"], capture_output=True, text=True, timeout=60)
                self._devices = p.stdout + p.stderr
            except (OSError, subprocess.TimeoutExpired) as e:
                self._devices = f"error: {e}"
        return self._devices

    _devices: Optional[str] = None

    def backend_detected(self) -> dict[int, bool]:
        if self.external:
            return {}
        pat = BACKEND_MARKERS[self.sc.engine]
        if self.sc.engine == "llamacpp":
            # The device list is authoritative; the log would also match paths such as build-nozendnn/.
            found = bool(pat.search(self.list_devices()))
            return {ip.idx: found for ip in self.instances}
        out = {}
        for ip in self.instances:
            try:
                text = ip.log_path.read_text(errors="replace")
            except OSError:
                text = ""
            engine_output = "\n".join(ln for ln in text.splitlines() if not ln.startswith(LAUNCH_HEADER))
            out[ip.idx] = bool(pat.search(engine_output))
        return out

    def backend_mismatch(self) -> Optional[str]:
        """Description of instances whose detected backend contradicts `accel`, or None."""
        want = self.sc.accel == "zendnn"
        bad = [i for i, found in self.backend_detected().items() if found != want]
        if not bad:
            return None
        what = "no ZenDNN/zentorch backend detected" if want else "ZenDNN/zentorch backend detected"
        return f"accel={self.sc.accel} but {what} on instance(s) {bad}"

    def launch_info(self) -> dict[str, Any]:
        relevant = re.compile(r"^(OMP_|KMP_|GOMP_|VLLM_|ZENDNN|ZENTORCH|TORCH|LD_|MALLOC|GGML_|LLAMA_"
                              r"|HF_|TRANSFORMERS_|PYTORCH_|PATH$|PYTHON)")
        return {
            "server_config_id": self.sc.id, "engine": self.sc.engine, "precision": self.sc.precision,
            "mode": "attach" if self.external else "launch",
            "accel": self.sc.accel, "backend_mismatch": self.backend_mismatch(),
            "pinning_method": self.pinning_method, "warnings": self.warnings,
            "inherited_env": {k: v for k, v in os.environ.items() if relevant.match(k)},
            "backend_detected": self.backend_detected(),
            "devices": self.list_devices(),
            "instances": [{
                "idx": ip.idx, "host": ip.host, "port": ip.port, "cores": ip.cores, "numa_node": ip.numa_node,
                "cmd": ip.cmd, "cmd_str": shlex.join(ip.cmd), "env_overrides": ip.env_overrides,
                "pid": ip.proc.pid if ip.proc else None, "ready_s": ip.ready_s,
                "model_name": ip.model_name, "log": str(ip.log_path),
            } for ip in self.instances],
        }

    def write_launch(self, extra: Optional[dict[str, Any]] = None) -> None:
        path = self.out_dir / "launch.json"
        history = []
        if path.exists():
            try:
                history = json.loads(path.read_text()).get("history", [])
            except ValueError:
                pass
        info = {**self.launch_info(), **(extra or {}), "launched_wall": time.time()}
        history.append(info)
        path.write_text(json.dumps({**info, "history": history}, indent=2, default=str))

    def extra_body(self) -> dict[str, Any]:
        """Per-engine request body additions."""
        if self.sc.engine == "llamacpp":
            return {"cache_prompt": False}
        return {}
