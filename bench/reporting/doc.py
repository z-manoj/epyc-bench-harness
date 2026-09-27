"""Report document model rendered to Markdown and a single self-contained HTML file."""

from __future__ import annotations

import base64
import html
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


class Doc:
    """Collects report blocks and renders them as Markdown and HTML."""

    def __init__(self, out_dir: Path) -> None:
        self.out = out_dir
        self.blocks: list[tuple[str, Any]] = []

    def h(self, level: int, text: str) -> None:
        self.blocks.append(("h", (level, text)))

    def p(self, text: str) -> None:
        self.blocks.append(("p", text))

    def table(self, df: pd.DataFrame, highlight: Optional[pd.DataFrame] = None) -> None:
        self.blocks.append(("table", (df, highlight)))

    def img(self, path: Path, alt: str) -> None:
        self.blocks.append(("img", (path, alt)))

    def markdown(self) -> str:
        out = []
        for kind, v in self.blocks:
            if kind == "h":
                out.append("#" * v[0] + " " + v[1])
            elif kind == "p":
                out.append(v)
            elif kind == "img":
                out.append(f"![{v[1]}]({v[0].relative_to(self.out)})")
            elif kind == "table":
                out.append(_md_table(*v))
            out.append("")
        return "\n".join(out)

    def html(self) -> str:
        parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Benchmark report</title><style>",
                 "body{font-family:sans-serif;max-width:1400px;margin:auto;padding:1em}"
                 "table{border-collapse:collapse;font-size:12px;margin:0.5em 0}"
                 "td,th{border:1px solid #ccc;padding:2px 6px;text-align:right}"
                 "th{background:#f0f0f0}td.bad{background:#f8c0c0;font-weight:bold}"
                 "img{max-width:100%}", "</style></head><body>"]
        for kind, v in self.blocks:
            if kind == "h":
                parts.append(f"<h{v[0]}>{html.escape(v[1])}</h{v[0]}>")
            elif kind == "p":
                parts.append(f"<p>{html.escape(v)}</p>")
            elif kind == "img":
                data = base64.b64encode(v[0].read_bytes()).decode()
                parts.append(f"<img alt='{html.escape(v[1])}' src='data:image/png;base64,{data}'/>")
            elif kind == "table":
                parts.append(_html_table(*v))
        parts.append("</body></html>")
        return "\n".join(parts)


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        a = abs(v)
        return f"{v:.3g}" if a < 100 else f"{v:.1f}" if a < 1e4 else f"{v:.0f}"
    return str(v)


def _md_table(df: pd.DataFrame, hl: Optional[pd.DataFrame]) -> str:
    if df.empty:
        return "_(none)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for i in range(len(df)):
        cells = []
        for j, c in enumerate(cols):
            s = _fmt(df.iat[i, j]).replace("|", "\\|")
            if hl is not None and c in hl.columns and bool(hl.iat[i, hl.columns.get_loc(c)]):
                s = f"**{s} (!)**"
            cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _html_table(df: pd.DataFrame, hl: Optional[pd.DataFrame]) -> str:
    if df.empty:
        return "<p><i>(none)</i></p>"
    cols = list(df.columns)
    rows = ["<table><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in cols) + "</tr>"]
    for i in range(len(df)):
        tds = []
        for j, c in enumerate(cols):
            bad = hl is not None and c in hl.columns and bool(hl.iat[i, hl.columns.get_loc(c)])
            tds.append(f"<td{' class=bad' if bad else ''}>{html.escape(_fmt(df.iat[i, j]))}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return "\n".join(rows) + "</table>"


