from __future__ import annotations

import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]+")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        obj = json.loads(text)
        if not isinstance(obj, list):
            raise ValueError(f"Expected a JSON array in {path}, got {type(obj).__name__}")
        return [x for x in obj if isinstance(x, dict)]

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                expanded = False
                for key in ("data", "records", "examples", "validation", "dev"):
                    value = item.get(key)
                    if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                        records.extend(value)
                        expanded = True
                        break
                if not expanded:
                    records.append(item)
            elif isinstance(item, list):
                records.extend(x for x in item if isinstance(x, dict))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]


def token_count(text: str) -> int:
    return len(tokenize(text))


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    toks = tokenize(text)
    if len(toks) <= chunk_size:
        return [text]
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(toks), step):
        end = min(len(toks), start + chunk_size)
        chunks.append(" ".join(toks[start:end]))
        if end >= len(toks):
            break
    return chunks


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", text)
    return " ".join(text.split())


def cosine_dict(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def bow(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for tok in tokenize(text):
        out[tok] = out.get(tok, 0.0) + 1.0
    return out


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.elapsed = time.perf_counter() - self.start


def top_items(scores: dict[str, float], k: int) -> list[str]:
    return [x for x, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]
