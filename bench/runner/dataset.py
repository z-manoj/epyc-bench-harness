"""Synthetic prompt dataset: <dataset_dir>/prompts_<size>.jsonl (or .txt), one JSON record per line
with an "id" and the prompt under "prompt" (or "text")."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    nominal_tokens: Optional[int]   # None when unknown (mixed file without a size field)


class DatasetError(RuntimeError):
    pass


def _load_file(path: Path, nominal: Optional[int]) -> list[Prompt]:
    out = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                pid = str(rec["id"])
                text = rec["prompt"] if "prompt" in rec else rec["text"]
            except (ValueError, KeyError) as e:
                raise DatasetError(f"{path}:{ln}: invalid record ({e})") from e
            if not isinstance(text, str) or not text:
                raise DatasetError(f"{path}:{ln}: empty prompt")
            nom = nominal
            for key in ("tokens", "size", "prompt_tokens", "nominal_tokens"):
                if key in rec:
                    nom = int(rec[key])
                    break
            out.append(Prompt(pid, text, nom))
    if not out:
        raise DatasetError(f"{path}: no prompts")
    return out


class Dataset:
    def __init__(self, root: Union[str, Path], sizes: Optional[list[int]] = None) -> None:
        self.root = Path(root)
        self.sizes = sizes or []
        self._cache: dict[Union[int, str], list[Prompt]] = {}

    def path(self, size: Union[int, str]) -> Optional[Path]:
        for ext in ("jsonl", "txt"):
            p = self.root / f"prompts_{size}.{ext}"
            if p.exists():
                return p
        return None

    def prompts(self, size: Union[int, str]) -> list[Prompt]:
        if size in self._cache:
            return self._cache[size]
        path = self.path(size)
        if path is not None:
            prompts = _load_file(path, None if size == "mixed" else int(size))
        elif size == "mixed":
            prompts = self._build_mixed()
        else:
            raise DatasetError(f"missing dataset file {self.root}/prompts_{size}.jsonl (or .txt)")
        self._cache[size] = prompts
        return prompts

    def _build_mixed(self) -> list[Prompt]:
        """Deterministic round-robin interleave of all numeric buckets."""
        sizes = sorted(self.sizes)
        buckets = [self.prompts(s) for s in sizes]
        if not buckets:
            raise DatasetError("mixed prompt size requested but no numeric sizes available")
        out: list[Prompt] = []
        for i in range(max(len(b) for b in buckets)):
            for s, b in zip(sizes, buckets):
                if i < len(b):
                    out.append(Prompt(f"{s}:{b[i].id}", b[i].text, b[i].nominal_tokens))
        return out

    def split(self, size: Union[int, str]) -> tuple[list[Prompt], list[Prompt]]:
        """(measurement pool, warmup pool): disjoint, deterministic."""
        prompts = self.prompts(size)
        if len(prompts) < 2:
            raise DatasetError(f"prompt size {size}: need >= 2 prompts for disjoint warmup/measurement")
        n_warm = max(1, min(len(prompts) // 4, 64))
        return prompts[: len(prompts) - n_warm], prompts[len(prompts) - n_warm:]
