# Artifact Audit Summary

Audit date: 2026-06-04.

## Current Status

- Local staging root: `artifacts/hkvm_rag_icde2027_artifact`.
- Public code uses paper-facing names instead of historical experiment phase names.
- Main verification command: `python scripts/verify_paper_evidence.py`.
- Data and frozen outputs are staged for Hugging Face Dataset hosting.

## Hosting Plan

| Platform | Contents |
|---|---|
| GitHub | `hkvm_mvp/`, `scripts/`, `configs/`, `docs/`, `results/`, `manifests/`, `README.md`, `LICENSE`, etc. |
| Hugging Face | `data/` (benchmarks + DeepSeek caches), `frozen_outputs/` (frozen predictions) |

`manifests/FILES.sha256` covers both parts; reviewers download from both
platforms and verify with `shasum -a 256 -c manifests/FILES.sha256`.

## Included

| Class | Location |
|---|---|
| Core code | `hkvm_mvp/` |
| Public configs | `configs/` |
| Public scripts | `scripts/` |
| Benchmark data | `data/benchmarks/` |
| DeepSeek caches | `data/llm_extraction_caches/` |
| Frozen predictions | `frozen_outputs/` |
| Paper evidence | `results/paper_evidence/` |
| Run summaries | `results/run_summaries/` |

## Required Checks

- `python scripts/verify_paper_evidence.py` should report `checks=63 bad=0`.
- `shasum -a 256 -c manifests/FILES.sha256` should pass after the release manifest is rebuilt.
- No `.DS_Store`, `__pycache__`, `.pyc`, backup, corrupt, smoke, dry-run, or temporary files should remain.
- No real API key or credential file should be present. `.env.example` contains only placeholders.

