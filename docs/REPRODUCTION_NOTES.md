# Reproduction Notes

## Prerequisites: Download Data from Hugging Face

The GitHub repository contains only code, configs, scripts, docs, results, and
manifests. `data/` and `frozen_outputs/` must be downloaded from the linked
Hugging Face Dataset repository and placed at the repository root. After
downloading, the layout should be:

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

Verify integrity:

```bash
shasum -a 256 -c manifests/FILES.sha256
```

## Level 1: Table Verification From Frozen Outputs

Run:

```bash
python scripts/verify_paper_evidence.py
```

This verifies the paper-facing fixed-substrate, adaptive evidence-controller,
source-robustness, bootstrap, and frozen-output coverage checks.

## Level 2: Regeneration From Processed Data And Caches

Use:

- `hkvm_mvp/`
- `configs/`
- `scripts/reproduce_fixed_substrate.py`
- `data/benchmarks/`
- `data/llm_extraction_caches/`

Example smoke test:

```bash
python scripts/reproduce_fixed_substrate.py \
  --setting 2wiki \
  --methods bm25,kg_ppr,weighted_hg_kv \
  --runs 1 \
  --limit 20 \
  --output_dir runs/smoke_2wiki
```

Changing the LLM endpoint, prompt, schema, or extraction budget requires
regenerating extraction caches and recording new cache hashes. Dense retrieval
reruns require downloading the upstream Contriever or ColBERTv2 model weights;
model weights are not included in this package.

