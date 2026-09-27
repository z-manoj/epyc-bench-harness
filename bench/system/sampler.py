"""50 ms CPU / memory sampler process (``python -m bench.system.sampler``).

Pinned to the harness cores, it uses deadline scheduling: sample k is taken at t0 + k * interval.
A deadline missed by more than one interval is counted in ``missed`` (the sample indices are
skipped, never back-filled).

On-disk format while running: a sequence of frames ``<u64 little-endian length><Arrow IPC stream>``
flushed roughly every second, so a crash loses at most ~1 s and the reader
(``bench.system.samples``) can tail the file.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa

CPU_FIELDS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
MEM_KEYS = {
    "MemTotal": "mem_total_mb", "MemAvailable": "mem_available_mb", "MemFree": "mem_free_mb",
    "Cached": "mem_cached_mb", "Buffers": "mem_buffers_mb", "AnonHugePages": "anon_hugepages_mb",
}
TREE_FIELDS = ("rss_mb", "pss_mb", "utime_s", "stime_s", "threads", "minflt", "majflt", "nprocs")
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
CLK_TCK = os.sysconf("SC_CLK_TCK")


def _schema() -> pa.Schema:
    f32l = pa.list_(pa.float32())
    f64l = pa.list_(pa.float64())
    fields = [
        pa.field("t_ns", pa.int64()), pa.field("sample_idx", pa.int64()),
        pa.field("lag_ns", pa.int64()), pa.field("missed_total", pa.int64()),
    ]
    fields += [pa.field(f"core_{f}", f32l) for f in CPU_FIELDS]
    fields += [pa.field("core_freq_mhz", f32l), pa.field("freq_fresh", pa.bool_())]
    fields += [pa.field(v, pa.float64()) for v in MEM_KEYS.values()]
    fields += [pa.field("mem_used_mb", pa.float64()), pa.field("hugepages_free", pa.float64())]
    fields += [pa.field("numa_used_mb", f64l), pa.field("numa_free_mb", f64l)]
    fields += [pa.field(f"tree_{f}", f64l) for f in TREE_FIELDS]
    return pa.schema(fields)


SCHEMA = _schema()


# ---------------------------------------------------------------------------
# /proc and /sys readers


def _pread(fd: int, size: int = 1 << 20) -> bytes:
    return os.pread(fd, size, 0)


class ProcReaders:
    def __init__(self) -> None:
        self.stat_fd = os.open("/proc/stat", os.O_RDONLY)
        self.meminfo_fd = os.open("/proc/meminfo", os.O_RDONLY)
        self.cpu_ids = [cid for cid, _ in self._parse_stat(_pread(self.stat_fd))]
        self.cpu_pos = {c: i for i, c in enumerate(self.cpu_ids)}
        self.numa_nodes: list[int] = []
        self.numa_fds: list[int] = []
        node_root = Path("/sys/devices/system/node")
        if node_root.exists():
            for p in sorted(node_root.glob("node[0-9]*"), key=lambda p: int(p.name[4:])):
                try:
                    self.numa_fds.append(os.open(p / "meminfo", os.O_RDONLY))
                    self.numa_nodes.append(int(p.name[4:]))
                except OSError:
                    pass
        self.freq_fds: list[int] = []
        for c in self.cpu_ids:
            path = f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq"
            try:
                self.freq_fds.append(os.open(path, os.O_RDONLY))
            except OSError:
                self.freq_fds = []
                break
        self.freq_source = "sysfs" if self.freq_fds else "cpuinfo"

    @staticmethod
    def _parse_stat(data: bytes) -> list[tuple[int, list[int]]]:
        out = []
        for line in data.split(b"\n"):
            if line.startswith(b"cpu") and len(line) > 3 and line[3:4].isdigit():
                parts = line.split()
                out.append((int(parts[0][3:]), [int(x) for x in parts[1:9]]))
        return out

    def cpu_jiffies(self) -> np.ndarray:
        rows = self._parse_stat(_pread(self.stat_fd))
        arr = np.zeros((len(self.cpu_ids), 8), dtype=np.int64)
        for cid, vals in rows:
            pos = self.cpu_pos.get(cid)
            if pos is not None:
                arr[pos, :len(vals)] = vals
        return arr

    def freqs_sysfs(self) -> np.ndarray:
        out = np.empty(len(self.freq_fds), dtype=np.float32)
        for i, fd in enumerate(self.freq_fds):
            try:
                out[i] = int(_pread(fd, 64)) / 1000.0
            except (OSError, ValueError):
                out[i] = np.nan
        return out

    def freqs_cpuinfo(self) -> np.ndarray:
        out = np.full(len(self.cpu_ids), np.nan, dtype=np.float32)
        cur = None
        with open("/proc/cpuinfo", "rb") as f:
            for line in f:
                if line.startswith(b"processor"):
                    cur = self.cpu_pos.get(int(line.split(b":")[1]))
                elif line.startswith(b"cpu MHz") and cur is not None:
                    out[cur] = float(line.split(b":")[1])
        return out

    def meminfo(self) -> dict[str, float]:
        vals: dict[str, float] = {}
        for line in _pread(self.meminfo_fd).split(b"\n"):
            if not line:
                continue
            k, _, rest = line.partition(b":")
            key = k.decode()
            if key in MEM_KEYS or key == "HugePages_Free":
                num = float(rest.split()[0])
                vals[key] = num if key == "HugePages_Free" else num / 1024.0
        out = {MEM_KEYS[k]: vals.get(k, np.nan) for k in MEM_KEYS}
        out["mem_used_mb"] = out["mem_total_mb"] - out["mem_available_mb"]
        out["hugepages_free"] = vals.get("HugePages_Free", np.nan)
        return out

    def numa(self) -> tuple[list[float], list[float]]:
        used, free = [], []
        for fd in self.numa_fds:
            u = f = np.nan
            for line in _pread(fd).split(b"\n"):
                if b"MemUsed:" in line:
                    u = float(line.split()[3]) / 1024.0
                elif b"MemFree:" in line:
                    f = float(line.split()[3]) / 1024.0
            used.append(u)
            free.append(f)
        return used, free


def read_pid_stat(pid: int) -> Optional[tuple[float, float, float, float, float, float]]:
    """(rss_mb, utime_s, stime_s, threads, minflt, majflt) or None if the process is gone."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read()
    except OSError:
        return None
    rest = data[data.rfind(b")") + 2:].split()
    try:
        return (int(rest[21]) * PAGE_SIZE / 2**20, int(rest[11]) / CLK_TCK, int(rest[12]) / CLK_TCK,
                float(rest[17]), float(rest[7]), float(rest[9]))
    except (IndexError, ValueError):
        return None


def read_pss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/smaps_rollup", "rb") as f:
            for line in f:
                if line.startswith(b"Pss:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def process_tree(root: int) -> list[int]:
    import psutil
    try:
        p = psutil.Process(root)
        return [root] + [c.pid for c in p.children(recursive=True)]
    except psutil.Error:
        return []


# ---------------------------------------------------------------------------
# Sampler process


class Sampler:
    def __init__(self, out: Path, interval_ms: float, ctl: Optional[Path], duration_s: Optional[float],
                 flush_s: float = 1.0) -> None:
        self.out = out
        self.iv = int(interval_ms * 1e6)
        self.ctl = ctl
        self.duration_s = duration_s
        self.flush_s = flush_s
        self.rd = ProcReaders()
        self.stop = threading.Event()
        self.roots: list[int] = []
        self.trees: list[list[int]] = []
        self.pss: list[float] = []
        self.pss_fresh = False
        self.freq = np.full(len(self.rd.cpu_ids), np.nan, dtype=np.float32)
        self.freq_fresh = False
        self.ctl_mtime = 0.0
        self.parent = os.getppid()
        self.missed = 0
        self.samples = 0

    # -- auxiliary thread: pid refresh (1 s), PSS (1 s), cpuinfo frequency (250 ms)
    def _aux(self) -> None:
        last_tree = last_pss = 0.0
        while not self.stop.is_set():
            now = time.monotonic()
            if self.rd.freq_source == "cpuinfo":
                try:
                    self.freq = self.rd.freqs_cpuinfo()
                    self.freq_fresh = True
                except OSError:
                    pass
            if now - last_tree >= 1.0:
                last_tree = now
                self._read_ctl()
                self.trees = [process_tree(r) for r in self.roots]
                if os.getppid() != self.parent:
                    self.stop.set()   # harness died
            if now - last_pss >= 1.0:
                last_pss = now
                self.pss = [float(np.nansum([read_pss_mb(p) for p in t])) if t else float("nan")
                            for t in self.trees]
                self.pss_fresh = True
            self.stop.wait(0.25)

    def _read_ctl(self) -> None:
        if not self.ctl:
            return
        try:
            m = self.ctl.stat().st_mtime
            if m == self.ctl_mtime:
                return
            d = json.loads(self.ctl.read_text())
            self.ctl_mtime = m
            self.roots = [int(p) for p in d.get("roots", [])]
            if d.get("stop"):
                self.stop.set()
        except (OSError, ValueError):
            pass

    def _cols(self) -> dict[str, list]:
        return {f.name: [] for f in SCHEMA}

    def run(self) -> None:
        self._read_ctl()
        self.trees = [process_tree(r) for r in self.roots]
        aux = threading.Thread(target=self._aux, daemon=True)
        aux.start()

        meta = {
            "cpu_ids": self.rd.cpu_ids, "numa_nodes": self.rd.numa_nodes,
            "interval_ms": self.iv / 1e6, "freq_source": self.rd.freq_source,
            "sampler_pid": os.getpid(), "start_wall": time.time(),
        }
        Path(str(self.out) + ".meta.json").write_text(json.dumps(meta))

        fh = open(self.out, "ab", buffering=0)
        # Prime the lazily-initialised IPC / zstd code paths so the first flush doesn't miss deadlines.
        cols = self._cols()
        self._sample(cols, time.monotonic_ns(), -1, 0, self.rd.cpu_jiffies())
        self._flush(open(os.devnull, "wb"), cols)
        self.samples = 0
        cols = self._cols()
        prev = self.rd.cpu_jiffies()
        t0 = time.monotonic_ns() + self.iv
        cpu0 = os.times()
        t_end = t0 + int(self.duration_s * 1e9) if self.duration_s else None
        last_flush = t0
        k = 0
        try:
            while not self.stop.is_set():
                deadline = t0 + k * self.iv
                now = time.monotonic_ns()
                if deadline > now:
                    time.sleep((deadline - now) / 1e9)
                    now = time.monotonic_ns()
                late = now - deadline
                if late > self.iv:
                    skip = late // self.iv
                    self.missed += skip
                    k += skip
                    late = now - (t0 + k * self.iv)
                if t_end and now >= t_end:
                    break
                prev = self._sample(cols, now, k, late, prev)
                k += 1
                if now - last_flush >= self.flush_s * 1e9:
                    self._flush(fh, cols)
                    cols = self._cols()
                    last_flush = now
        finally:
            self._flush(fh, cols)
            fh.close()
            self.stop.set()
            ru = os.times()
            Path(str(self.out) + ".final.json").write_text(json.dumps({
                "samples": self.samples, "missed": self.missed,
                "loop_cpu_s": (ru.user + ru.system) - (cpu0.user + cpu0.system),
                "loop_wall_s": (time.monotonic_ns() - t0) / 1e9,
                "self_cpu_s": ru.user + ru.system, "end_wall": time.time(),
            }))

    def _sample(self, cols: dict[str, list], now: int, k: int, late: int, prev: np.ndarray) -> np.ndarray:
        cur = self.rd.cpu_jiffies()
        d = (cur - prev).astype(np.float64)
        tot = d.sum(axis=1)
        tot[tot <= 0] = np.nan
        pct = (d / tot[:, None] * 100.0).astype(np.float32)
        cols["t_ns"].append(now)
        cols["sample_idx"].append(k)
        cols["lag_ns"].append(late)
        cols["missed_total"].append(self.missed)
        for i, f in enumerate(CPU_FIELDS):
            cols[f"core_{f}"].append(pct[:, i])
        if self.rd.freq_source == "sysfs":
            cols["core_freq_mhz"].append(self.rd.freqs_sysfs())
            cols["freq_fresh"].append(True)
        else:
            cols["core_freq_mhz"].append(self.freq)
            cols["freq_fresh"].append(self.freq_fresh)
            self.freq_fresh = False
        mem = self.rd.meminfo()
        for key, v in mem.items():
            cols[key].append(v)
        nu, nf = self.rd.numa()
        cols["numa_used_mb"].append(nu)
        cols["numa_free_mb"].append(nf)

        trees = self.trees
        agg = np.zeros((len(trees), len(TREE_FIELDS)), dtype=np.float64)
        for ti, pids in enumerate(trees):
            for pid in pids:
                st = read_pid_stat(pid)
                if st is None:
                    continue
                rss, ut, stt, thr, mnf, mjf = st
                agg[ti, 0] += rss
                agg[ti, 2] += ut
                agg[ti, 3] += stt
                agg[ti, 4] += thr
                agg[ti, 5] += mnf
                agg[ti, 6] += mjf
                agg[ti, 7] += 1
            agg[ti, 1] = self.pss[ti] if (self.pss_fresh and ti < len(self.pss)) else np.nan
        self.pss_fresh = False
        for i, f in enumerate(TREE_FIELDS):
            cols[f"tree_{f}"].append(agg[:, i].tolist())
        self.samples += 1
        return cur

    def _flush(self, fh, cols: dict[str, list]) -> None:
        if not cols["t_ns"]:
            return
        arrays = []
        for f in SCHEMA:
            v = cols[f.name]
            if pa.types.is_list(f.type) and f.type.value_type == pa.float32():
                arrays.append(pa.array([np.asarray(x, dtype=np.float32) for x in v], type=f.type))
            else:
                arrays.append(pa.array(v, type=f.type))
        batch = pa.RecordBatch.from_arrays(arrays, schema=SCHEMA)
        sink = pa.BufferOutputStream()
        opts = pa.ipc.IpcWriteOptions(compression="zstd")
        with pa.ipc.new_stream(sink, SCHEMA, options=opts) as w:
            w.write_batch(batch)
        buf = sink.getvalue().to_pybytes()
        fh.write(struct.pack("<Q", len(buf)) + buf)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="bench.system.sampler")
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval-ms", type=float, default=50)
    ap.add_argument("--ctl", default=None, help="JSON control file: {roots: [pid...], stop: bool}")
    ap.add_argument("--cores", default=None, help="cpulist to pin to")
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--nice", type=int, default=-10)
    a = ap.parse_args(argv)
    if a.cores:
        from bench.core.config import parse_cores
        try:
            os.sched_setaffinity(0, parse_cores(a.cores))
        except OSError as e:
            print(f"sampler: cannot pin to {a.cores}: {e}", file=sys.stderr)
    try:
        os.nice(a.nice)
    except (PermissionError, OSError):
        pass
    s = Sampler(Path(a.out), a.interval_ms, Path(a.ctl) if a.ctl else None, a.duration_s)
    signal.signal(signal.SIGTERM, lambda *_: s.stop.set())
    signal.signal(signal.SIGINT, lambda *_: s.stop.set())
    s.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
