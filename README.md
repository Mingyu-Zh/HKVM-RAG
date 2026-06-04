# HKVM-RAG ICDE 2027 Reproducibility Artifact

This repository contains the code, staged data artifacts, frozen outputs, and
paper-facing evidence files for the HKVM-RAG ICDE 2027 submission.

The public artifact is organized around the paper's scientific claims, not the
historical experiment phases used during development.

## Artifact Boundary

The repository supports two reproducibility modes.

1. **Paper evidence verification.** Use `results/paper_evidence` and
   `frozen_outputs` to verify the exact reported tables, bootstrap intervals,
   and claim-evidence maps.
2. **Method regeneration.** Use `hkvm_mvp`, `configs`, `scripts`, and
   `data/benchmarks` plus `data/llm_extraction_caches` to rerun the
   fixed-substrate HKVM-RAG key-space comparison without issuing new LLM
   extraction calls.

The artifact does not include raw upstream dataset archives, model weights,
Hugging Face caches, private API keys, remote run directories, or historical
debug/smoke runs.

## Layout

| Path | Purpose |
|---|---|
| `hkvm_mvp/` | Core HKVM-RAG implementation. |
| `configs/` | Reviewer-facing configs named by paper concept. |
| `scripts/reproduce_fixed_substrate.py` | Entry point for fixed-substrate reruns. |
| `scripts/verify_paper_evidence.py` | Deterministic verification of paper-facing results. |
| `data/benchmarks/` | Normalized benchmark splits used in the paper. |
| `data/llm_extraction_caches/` | DeepSeek extraction caches and sidecar manifests. |
| `frozen_outputs/` | Frozen prediction files used to audit reported results. |
| `results/paper_evidence/` | Tables, bootstrap summaries, and claim-evidence maps. |
| `results/run_summaries/` | Compact run summaries retained for provenance. |
| `docs/` | Release, attribution, and reproduction notes. |
| `manifests/` | SHA-256 file manifest and release manifest. |

## Quick Verification

```bash
python scripts/verify_paper_evidence.py
```

Expected output:

```text
checks=63 bad=0
```

## Fixed-Substrate Rerun

Create the environment:

```bash
conda env create -f environment.yml
conda activate hkvm
```

Run a small smoke test:

```bash
python scripts/reproduce_fixed_substrate.py \
  --setting 2wiki \
  --methods bm25,kg_ppr,weighted_hg_kv \
  --runs 1 \
  --limit 20 \
  --output_dir runs/smoke_2wiki
```

The full fixed-substrate reruns use the same entry point with `--setting
2wiki`, `musique`, or `hotpotqa`.

## Hosting Notes

This artifact is split across two hosting platforms:

| Platform | Contents | Purpose |
|---|---|---|
| **GitHub** | `hkvm_mvp/`, `scripts/`, `configs/`, `docs/`, `results/`, `manifests/`, `README.md`, `LICENSE`, `NOTICE.md`, `requirements.txt`, `environment.yml`, `.env.example`, `.gitignore` | Source code, public entry points, paper evidence tables, file manifests, and documentation. |
| **Hugging Face** | `data/`, `frozen_outputs/` | Large reproducibility data: benchmark splits, DeepSeek LLM extraction caches, and frozen prediction files. |

### Setup

1. Clone the GitHub repository.
2. Download `data/` and `frozen_outputs/` from the linked Hugging Face Dataset
   repository and place them at the repository root, preserving the directory
   layout. The final layout must be:
   ```
   <repo_root>/
   ├── data/
   │   ├── benchmarks/
   │   └── llm_extraction_caches/
   ├── frozen_outputs/
   │   ├── manifests/
   │   └── predictions/
   └── ...
   ```
3. Verify:
   ```bash
   python scripts/verify_paper_evidence.py
   shasum -a 256 -c manifests/FILES.sha256
   ```

The `manifests/FILES.sha256` file covers the complete artifact (GitHub + Hugging
Face) and can be used to verify file integrity after downloading both parts.

