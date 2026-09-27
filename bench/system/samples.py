"""Reader side of the sampler: tail the live frame file (``SampleStore``), convert sessions to
``samples.parquet`` and expose rows as numpy arrays (``Samples``) for gates, validation and plots."""

from __future__ import annotations

import asyncio
import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bench.system.sampler import CPU_FIELDS, MEM_KEYS, SCHEMA, TREE_FIELDS


def iter_frames(path: Path, offset: int = 0) -> tuple[list[pa.RecordBatch], int]:
    """Read complete frames starting at offset.  Returns (batches, new_offset)."""
    batches: list[pa.RecordBatch] = []
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                (n,) = struct.unpack("<Q", hdr)
                body = f.read(n)
                if len(body) < n:
                    break   # partial frame (writer mid-flush or crashed): stop here
                try:
                    with pa.ipc.open_stream(pa.py_buffer(body)) as r:
                        batches.extend(r)
                except pa.ArrowInvalid:
                    break
                offset += 8 + n
    except FileNotFoundError:
        pass
    return batches, offset


@dataclass
class Samples:
    """Columnar view of sampler rows as numpy arrays."""
    cpu_ids: list[int]
    interval_ms: float
    t_ns: np.ndarray
    sample_idx: np.ndarray
    lag_ns: np.ndarray
    cpu: dict[str, np.ndarray]           # field -> (n, ncpu) percent
    freq: np.ndarray                     # (n, ncpu) MHz
    mem: dict[str, np.ndarray]           # scalar memory series
    numa_used: np.ndarray                # (n, nodes)
    tree: dict[str, np.ndarray]          # field -> (n, ntrees)
    numa_nodes: list[int]

    @property
    def n(self) -> int:
        return len(self.t_ns)

    def pos(self, cores: Iterable[int]) -> list[int]:
        m = {c: i for i, c in enumerate(self.cpu_ids)}
        return [m[c] for c in cores if c in m]

    def slice(self, t0: int, t1: int) -> "Samples":
        sel = (self.t_ns >= t0) & (self.t_ns <= t1)
        return Samples(
            self.cpu_ids, self.interval_ms, self.t_ns[sel], self.sample_idx[sel], self.lag_ns[sel],
            {k: v[sel] for k, v in self.cpu.items()}, self.freq[sel],
            {k: v[sel] for k, v in self.mem.items()}, self.numa_used[sel],
            {k: v[sel] for k, v in self.tree.items()}, self.numa_nodes)

    def busy(self) -> np.ndarray:
        """(n, ncpu) utilisation = 100 - idle - iowait."""
        return 100.0 - self.cpu["idle"] - self.cpu["iowait"]

    def util(self, cores: Iterable[int], kind: str = "busy") -> np.ndarray:
        """Per-sample mean utilisation over the given cores. kind in busy|user|system."""
        p = self.pos(cores)
        if not p or self.n == 0:
            return np.full(self.n, np.nan)
        if kind == "busy":
            m = self.busy()[:, p]
        elif kind == "user":
            m = self.cpu["user"][:, p] + self.cpu["nice"][:, p]
        elif kind == "system":
            m = self.cpu["system"][:, p] + self.cpu["irq"][:, p] + self.cpu["softirq"][:, p]
        else:
            raise ValueError(kind)
        with np.errstate(all="ignore"):
            return np.nanmean(m, axis=1)

    def mean_freq(self, cores: Iterable[int]) -> float:
        p = self.pos(cores)
        if not p or self.n == 0:
            return float("nan")
        v = self.freq[:, p]
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")

    def rss_total(self) -> np.ndarray:
        r = self.tree.get("rss_mb")
        if r is None or r.shape[1] == 0:
            return np.full(self.n, np.nan)
        return np.nansum(r, axis=1)

    def pss_total(self) -> np.ndarray:
        r = self.tree.get("pss_mb")
        if r is None or r.shape[1] == 0:
            return np.full(self.n, np.nan)
        with np.errstate(all="ignore"):
            out = np.nansum(r, axis=1)
        out[~np.isfinite(r).any(axis=1)] = np.nan
        return out

    def missed_frac(self) -> float:
        """Fraction of deadlines in the window without a sample."""
        if self.n < 2:
            return 1.0 if self.n == 0 else 0.0
        expected = int(self.sample_idx[-1] - self.sample_idx[0] + 1)
        return max(0.0, 1.0 - self.n / expected)


def _list2d(col: pa.ChunkedArray | pa.Array, width: Optional[int] = None) -> np.ndarray:
    rows = col.to_pylist()
    w = width if width is not None else max((len(r) for r in rows if r is not None), default=0)
    out = np.full((len(rows), w), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        if r:
            m = min(len(r), w)
            out[i, :m] = [np.nan if x is None else x for x in r[:m]]
    return out


def _fixed2d(col: pa.ChunkedArray, width: int) -> np.ndarray:
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.float64)
    if len(flat) == len(col) * width:
        return flat.reshape(len(col), width)
    return _list2d(col, width)


def table_to_samples(table: pa.Table, meta: dict) -> Samples:
    cpu_ids = meta["cpu_ids"]
    nc = len(cpu_ids)
    g = lambda name: table.column(name)  # noqa: E731
    return Samples(
        cpu_ids=cpu_ids, interval_ms=meta["interval_ms"],
        t_ns=g("t_ns").to_numpy(), sample_idx=g("sample_idx").to_numpy(), lag_ns=g("lag_ns").to_numpy(),
        cpu={f: _fixed2d(g(f"core_{f}"), nc) for f in CPU_FIELDS},
        freq=_fixed2d(g("core_freq_mhz"), nc),
        mem={k: g(k).to_numpy().astype(np.float64)
             for k in list(MEM_KEYS.values()) + ["mem_used_mb", "hugepages_free"]},
        numa_used=_list2d(g("numa_used_mb"), len(meta.get("numa_nodes", [])) or None),
        tree={f: _list2d(g(f"tree_{f}")) for f in TREE_FIELDS},
        numa_nodes=meta.get("numa_nodes", []),
    )


def empty_table() -> pa.Table:
    return SCHEMA.empty_table()


class SampleStore:
    """Incrementally tails the sampler frame file of the current session."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.batches: list[pa.RecordBatch] = []
        self.meta: Optional[dict] = None
        self.last_t = 0

    def _meta(self) -> Optional[dict]:
        if self.meta is None:
            p = Path(str(self.path) + ".meta.json")
            if p.exists():
                self.meta = json.loads(p.read_text())
        return self.meta

    def poll(self) -> None:
        new, self.offset = iter_frames(self.path, self.offset)
        if new:
            self.batches.extend(new)
            self.last_t = int(new[-1].column("t_ns")[-1].as_py())

    async def wait_for(self, t_ns: int, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while True:
            self.poll()
            if self.last_t >= t_ns:
                return True
            if time.monotonic() > deadline:
                return False
            await asyncio.sleep(0.2)

    def trim(self, before_t: int) -> None:
        self.batches = [b for b in self.batches if int(b.column("t_ns")[-1].as_py()) >= before_t]

    def window(self, t0: int, t1: int) -> Samples:
        self.poll()
        meta = self._meta()
        if meta is None:
            raise RuntimeError("sampler metadata not found; sampler not running?")
        sel = [b for b in self.batches
               if int(b.column("t_ns")[-1].as_py()) >= t0 and int(b.column("t_ns")[0].as_py()) <= t1]
        table = pa.Table.from_batches(sel, schema=SCHEMA) if sel else empty_table()
        return table_to_samples(table, meta).slice(t0, t1)


def finalize_samples(config_dir: Path) -> Optional[Path]:
    """Convert every sampler session frame file into Parquet and build the combined samples.parquet."""
    sdir = config_dir / "samples"
    if not sdir.exists():
        return None
    for bin_path in sorted(sdir.glob("session_*.bin")):
        pq_path = bin_path.with_suffix(".parquet")
        batches, _ = iter_frames(bin_path)
        table = pa.Table.from_batches(batches, schema=SCHEMA) if batches else empty_table()
        pq.write_table(table, pq_path, compression="zstd")
        bin_path.unlink()
    parts = sorted(sdir.glob("session_*.parquet"))
    if not parts:
        return None
    metas = [json.loads(Path(str(p.with_suffix(".bin")) + ".meta.json").read_text())
             for p in parts if Path(str(p.with_suffix(".bin")) + ".meta.json").exists()]
    table = pa.concat_tables([pq.read_table(p) for p in parts])
    table = table.sort_by("t_ns")
    out = config_dir / "samples.parquet"
    meta = metas[0] if metas else {}
    table = table.replace_schema_metadata({"bench_meta": json.dumps(meta)})
    pq.write_table(table, out, compression="zstd")
    return out


def load_samples(config_dir: Path) -> Optional[Samples]:
    """Load samples for a server config directory (finalizing leftover frame files first)."""
    sdir = config_dir / "samples"
    if sdir.exists() and any(sdir.glob("session_*.bin")):
        finalize_samples(config_dir)
    p = config_dir / "samples.parquet"
    if not p.exists():
        if sdir.exists() and any(sdir.glob("session_*.parquet")):
            finalize_samples(config_dir)
        if not p.exists():
            return None
    table = pq.read_table(p)
    md = table.schema.metadata or {}
    meta = json.loads(md.get(b"bench_meta", b"{}"))
    if not meta:
        return None
    return table_to_samples(table, meta)

