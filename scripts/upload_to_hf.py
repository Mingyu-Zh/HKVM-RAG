#!/usr/bin/env python3
"""Robust upload of data/ and frozen_outputs/ to Hugging Face Dataset.

Uploads files from smallest to largest with retry logic, resume support,
and Xet disabled for China network compatibility.

Usage:
    HF_HUB_ENABLE_HF_XET=0 python scripts/upload_to_hf.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "MingY-Zh/HKVM-RAG"
REPO_TYPE = "dataset"
MAX_RETRIES = 5
RETRY_BASE_DELAY = 5  # seconds, exponential backoff


def upload_file(api, path: Path, rel: str, repo_id: str) -> bool:
    """Upload a single file with retry logic. Returns True on success."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  [{rel}] uploading ({path.stat().st_size / 1e6:.1f} MB)...", end=" ", flush=True)
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=rel,
                repo_id=repo_id,
                repo_type=REPO_TYPE,
            )
            print("OK")
            return True
        except Exception as exc:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                print(f"FAILED (attempt {attempt}/{MAX_RETRIES}), retry in {delay}s: {exc}")
                time.sleep(delay)
            else:
                print(f"GIVING UP after {MAX_RETRIES} attempts: {exc}")
                return False
    return False


def upload_directory(api, root: Path, dir_name: str, repo_id: str) -> tuple[int, int]:
    """Upload all files in a directory from smallest to largest.
    Skips files that already exist remotely.
    Returns (uploaded, skipped_or_failed).
    """
    # Collect all files, exclude junk
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = set(path.parts)
        if {"__pycache__", ".git", ".cache", ".DS_Store"} & parts:
            continue
        if path.suffix in {".pyc", ".pyo"} or path.name == ".DS_Store":
            continue
        rel = path.relative_to(ROOT).as_posix()
        files.append((path, rel))

    # Sort by size: smallest first
    files.sort(key=lambda x: x[0].stat().st_size)

    print(f"\n{'='*60}")
    print(f"Uploading {dir_name}/ ({len(files)} files)")
    print(f"{'='*60}")

    # Check what's already on HF
    try:
        existing = {item.rfilename for item in api.list_repo_files(repo_id=repo_id, repo_type=REPO_TYPE)}
    except Exception:
        existing = set()
    print(f"  {len(existing)} files already on remote")

    uploaded, skipped = 0, 0
    for path, rel in files:
        if rel in existing:
            skipped += 1
            continue
        if upload_file(api, path, rel, repo_id):
            uploaded += 1
        else:
            skipped += 1
            print(f"\n  ERROR: Upload of {rel} failed. You can re-run this script to retry remaining files.")

    print(f"\n  Uploaded: {uploaded}, Skipped/Existing: {skipped}")
    return uploaded, skipped


def main() -> None:
    from huggingface_hub import HfApi

    api = HfApi()

    print("HF Hub upload script for HKVM-RAG artifact")
    print(f"Target: {REPO_ID} ({REPO_TYPE})")
    print(f"Xet enabled: {os.environ.get('HF_HUB_ENABLE_HF_XET', 'not set')}")

    dirs = ["data", "frozen_outputs"]
    total_up, total_skip = 0, 0
    for d in dirs:
        up, skip = upload_directory(api, ROOT / d, d, REPO_ID)
        total_up += up
        total_skip += skip

    print(f"\n{'='*60}")
    print(f"Done. Uploaded: {total_up}, Skipped/Existing/Failed: {total_skip}")
    print(f"HF Dataset: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
