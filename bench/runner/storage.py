"""On-disk rep state: terminal rep summaries, cleanup of interrupted reps, requests.parquet."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from bench.core import status as S


def rep_history(point_dir: Path) -> list[dict[str, Any]]:
    """Terminal rep summaries of a point, ordered by rep number."""
    hist = []
    if not point_dir.exists():
        return hist
    for d in point_dir.glob("rep_*"):
        p = d / "summary.json"
        if p.exists():
            try:
                s = json.loads(p.read_text())
            except ValueError:
                continue
            if s.get("status") in S.REP_STATUSES:
                hist.append(s)
    return sorted(hist, key=lambda s: s["rep"])


def clean_incomplete_reps(point_dir: Path) -> list[int]:
    """Remove rep directories without a terminal summary (interrupted reps) so they are re-run."""
    removed = []
    if not point_dir.exists():
        return removed
    for d in point_dir.glob("rep_*"):
        p = d / "summary.json"
        ok = False
        if p.exists():
            try:
                ok = json.loads(p.read_text()).get("status") in S.REP_STATUSES
            except ValueError:
                ok = False
        if not ok:
            shutil.rmtree(d)
            removed.append(int(d.name.split("_")[1]))
    return removed


def write_requests(path: Path, records: list[dict[str, Any]], extra: dict[str, Any]) -> None:
    rows = [{**extra, **r} for r in records]
    schema = pa.schema([
        ("request_id", pa.string()), ("phase", pa.string()), ("instance", pa.int32()),
        ("prompt_id", pa.string()), ("nominal_prompt_tokens", pa.int64()), ("output_len", pa.int64()),
        ("t_submit", pa.int64()), ("t_send", pa.int64()), ("t_first_token", pa.int64()),
        ("t_done", pa.int64()), ("chunk_ts", pa.list_(pa.int64())), ("n_chunks", pa.int64()),
        ("prompt_tokens", pa.int64()), ("completion_tokens", pa.int64()),
        ("completion_tokens_source", pa.string()), ("http_status", pa.int32()), ("error", pa.string()),
        *[(k, pa.string() if isinstance(v, str) else pa.int64()) for k, v in extra.items()],
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")

