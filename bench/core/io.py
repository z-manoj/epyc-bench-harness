"""Logging, atomic JSON writes and the events.jsonl phase-marker log."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Optional


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, obj: Any) -> None:
    """Atomic JSON write (tmp + rename), so readers never see partial files."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, phase: str, event: str, point_id: Optional[str] = None, rep: Optional[int] = None,
              **extra: Any) -> int:
        t = time.monotonic_ns()
        rec = {"t_ns": t, "wall": time.time(), "point_id": point_id, "rep": rep, "phase": phase,
               "event": event, **extra}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")
        return t


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


