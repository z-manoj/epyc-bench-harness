"""Platform + software metadata snapshot, and the per-server-config "did anything change" check."""

from __future__ import annotations

import hashlib
import json
import platform
import shlex
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

from bench.core.config import ExperimentConfig, ServerConfig


def _read(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _run(cmd: list[str], timeout: float = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + (("\n" + p.stderr) if p.stderr else "")).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, f"{type(e).__name__}: {e}"


def _bracketed(s: Optional[str]) -> Optional[str]:
    """'always [madvise] never' -> 'madvise'."""
    if s and "[" in s:
        return s[s.index("[") + 1: s.index("]")]
    return s


def tunables() -> dict[str, Any]:
    """Settings that must not change during an experiment."""
    cpu_root = Path("/sys/devices/system/cpu")
    governors: dict[str, str] = {}
    for p in sorted(cpu_root.glob("cpu[0-9]*/cpufreq/scaling_governor")):
        governors[p.parent.parent.name] = p.read_text().strip()
    gov_set = sorted(set(governors.values()))
    return {
        "smt_control": _read("/sys/devices/system/cpu/smt/control"),
        "smt_active": _read("/sys/devices/system/cpu/smt/active"),
        "governors": gov_set,                                       # distinct values across cores
        "thp_enabled": _bracketed(_read("/sys/kernel/mm/transparent_hugepage/enabled")),
        "thp_defrag": _bracketed(_read("/sys/kernel/mm/transparent_hugepage/defrag")),
        "numa_balancing": _read("/proc/sys/kernel/numa_balancing"),
        "online_cpus": _read("/sys/devices/system/cpu/online"),
    }


def cstates() -> dict[str, Any]:
    out: dict[str, Any] = {}
    root = Path("/sys/devices/system/cpu/cpu0/cpuidle")
    for st in sorted(root.glob("state*")):
        out[st.name] = {k: _read(str(st / k)) for k in ("name", "latency", "disable")}
    out["cpuidle_driver"] = _read("/sys/devices/system/cpu/cpuidle/current_driver")
    return out


def check_expected(cfg: ExperimentConfig, t: dict[str, Any]) -> list[str]:
    problems = []
    exp_smt = cfg.platform.expected_smt
    if exp_smt is not None:
        active = t.get("smt_active")
        actual = {"1": "on", "0": "off"}.get(active or "", t.get("smt_control"))
        if actual != exp_smt:
            problems.append(f"SMT is {actual!r}, expected {exp_smt!r}")
    gov = cfg.platform.expected_governor
    if gov is not None and t.get("governors") != [gov]:
        problems.append(f"CPU governor(s) {t.get('governors')}, expected {gov!r}")
    return problems


def diff_tunables(before: dict[str, Any], now: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    keys = ("smt_control", "smt_active", "governors", "thp_enabled", "thp_defrag", "numa_balancing")
    return {k: (before.get(k), now.get(k)) for k in keys if before.get(k) != now.get(k)}


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def model_checksums(sc: ServerConfig, cache: dict[str, Any]) -> dict[str, Any]:
    p = Path(sc.model)
    if sc.mode == "attach" and not p.exists():
        return {"served_model_name": sc.model}
    targets: list[Path] = []
    if p.is_file():
        targets = [p]
    elif p.is_dir():
        targets = [q for q in (p / "model.safetensors.index.json", p / "config.json") if q.exists()]
        if not targets:
            targets = sorted(p.glob("*.safetensors"))[:1]
    out: dict[str, Any] = {}
    for t in targets:
        st = t.stat()
        key = f"{t}|{st.st_size}|{int(st.st_mtime)}"
        if key not in cache:
            cache[key] = sha256_file(t)
        out[str(t)] = {"sha256": cache[key], "size": st.st_size}
    if not targets:
        out["_note"] = f"model path {sc.model} not found locally"
    return out


def _pkg_versions() -> dict[str, Optional[str]]:
    out = {}
    for name in ("torch", "vllm", "zentorch", "numpy", "aiohttp", "pandas", "pyarrow"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def engine_versions(sc: ServerConfig) -> dict[str, Any]:
    info: dict[str, Any] = {"engine": sc.engine, "mode": sc.mode}
    if sc.mode == "attach":
        info["endpoints"] = [f"http://{e.host}:{e.port}" for e in sc.endpoints]
        return info
    if sc.engine == "llamacpp" and sc.binary:
        rc, out = _run([sc.binary, "--version"], timeout=30)
        info["version_output"] = out[-4000:]
        src = Path(sc.binary).resolve()
        for parent in src.parents:
            if (parent / ".git").exists():
                info["git_commit"] = _run(["git", "-C", str(parent), "rev-parse", "HEAD"])[1]
                info["git_dirty"] = bool(_run(["git", "-C", str(parent), "status", "--porcelain"])[1])
                cache = parent / "build" / "CMakeCache.txt"
                if cache.exists():
                    flags = [ln for ln in cache.read_text(errors="replace").splitlines()
                             if ln.startswith(("GGML_", "CMAKE_BUILD_TYPE", "CMAKE_C_FLAGS", "LLAMA_"))
                             and not ln.startswith("//")]
                    info["build_flags"] = flags
                break
    elif sc.engine == "vllm":
        exe = shlex.split(sc.binary) if sc.binary else ["vllm"]
        info["version_output"] = _run([*exe, "--version"], timeout=120)[1][-2000:]
        info["python_pkgs"] = _run([_script_python(exe[0]), "-c", _VERSIONS_SNIPPET], timeout=120)[1]
    return info


_VERSIONS_SNIPPET = """
import importlib.metadata as m, json
out = {}
for p in ("torch", "vllm", "zentorch"):
    try:
        out[p] = m.version(p)
    except m.PackageNotFoundError:
        out[p] = None
try:
    import torch
    out["torch_config"] = torch.__config__.show()
except Exception as e:
    out["torch_config"] = repr(e)
print(json.dumps(out))
"""


def _script_python(exe: str) -> str:
    """Interpreter from a console script's shebang (so versions come from the engine's env)."""
    import shutil
    path = shutil.which(exe)
    if path:
        try:
            first = Path(path).read_text(errors="replace").splitlines()[0]
            if first.startswith("#!") and "python" in first:
                return first[2:].strip().split()[0]
        except (OSError, IndexError):
            pass
    return sys.executable


def capture(cfg: ExperimentConfig, env_dir: Path) -> dict[str, Any]:
    env_dir.mkdir(parents=True, exist_ok=True)
    cmds = {
        "lscpu.txt": ["lscpu"],
        "numactl_H.txt": ["numactl", "-H"],
        "uname.txt": ["uname", "-a"],
        "dmidecode.txt": ["dmidecode", "-t", "bios", "-t", "memory"],
        "pip_freeze.txt": [sys.executable, "-m", "pip", "freeze"],
    }
    for fname, cmd in cmds.items():
        rc, out = _run(cmd)
        if rc != 0 and fname == "dmidecode.txt":
            rc2, out2 = _run(["sudo", "-n", *cmd])
            if rc2 == 0:
                rc, out = rc2, out2
        (env_dir / fname).write_text(f"$ {shlex.join(cmd)}\n# rc={rc}\n{out}\n")

    cpuinfo = _read("/proc/cpuinfo") or ""
    (env_dir / "cpuinfo_core0.txt").write_text(cpuinfo.split("\n\n")[0] + "\n")
    (env_dir / "cmdline.txt").write_text((_read("/proc/cmdline") or "") + "\n")

    t = tunables()
    cache_path = env_dir / "checksum_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    servers = {}
    for sc in cfg.server_configs:
        servers[sc.id] = {"versions": engine_versions(sc), "model": model_checksums(sc, cache)}
    cache_path.write_text(json.dumps(cache, indent=1))

    snapshot = {
        "kernel": platform.release(),
        "python": sys.version,
        "hostname": platform.node(),
        "tunables": t,
        "cstates": cstates(),
        "expected_problems": check_expected(cfg, t),
        "packages": _pkg_versions(),
        "servers": servers,
    }
    (env_dir / "env.json").write_text(json.dumps(snapshot, indent=2, default=str))
    return snapshot
