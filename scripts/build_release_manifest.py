#!/usr/bin/env python3
"""Build package-level file checksums for the HKVM-RAG artifact."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "manifests"
EXCLUDE_REL = {
    "manifests/FILES.sha256",
    "manifests/artifact_release_manifest.json",
}
EXCLUDE_DIRS = {"__pycache__", ".git", ".cache", "runs", "logs", "tmp"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pyd"}
EXCLUDE_NAMES = {".DS_Store"}
EXCLUDE_PREFIXES = ("runs/", "logs/", "tmp/", ".cache/", ".git/")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_class(rel: str) -> str:
    if rel.startswith("code/"):
        return "code"
    if rel.startswith("data/benchmarks/"):
        return "benchmark_data"
    if rel.startswith("data/llm_extraction_caches/"):
        return "llm_cache"
    if rel.startswith("data/upstream/"):
        return "data_documentation"
    if rel.startswith("frozen_outputs/"):
        return "frozen_output"
    if rel.startswith("results/"):
        return "result"
    if rel.startswith("docs/"):
        return "documentation"
    if rel.startswith("scripts/"):
        return "packaging_script"
    return "other"


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        parts = set(path.parts)
        if EXCLUDE_DIRS & parts:
            continue
        if path.suffix in EXCLUDE_SUFFIXES or path.name in EXCLUDE_NAMES:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXCLUDE_REL:
            continue
        if rel.startswith(EXCLUDE_PREFIXES):
            continue
        digest = sha256(path)
        records.append(
            {
                "path": rel,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "class": file_class(rel),
            }
        )

    with (MANIFEST_DIR / "FILES.sha256").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(f"{r['sha256']}  {r['path']}\n")

    summary = {}
    for r in records:
        cls = r["class"]
        item = summary.setdefault(cls, {"file_count": 0, "size_bytes": 0})
        item["file_count"] += 1
        item["size_bytes"] += r["size_bytes"]

    manifest = {
        "artifact_release_manifest_version": "0.2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root_name": ROOT.name,
        "raw_datasets_included": False,
        "model_weights_included": False,
        "secrets_included": False,
        "summary": summary,
        "files": records,
    }
    (MANIFEST_DIR / "artifact_release_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
