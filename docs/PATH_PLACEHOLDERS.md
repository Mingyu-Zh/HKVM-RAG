# Path Conventions

Reviewer-facing configs use artifact-relative paths:

- Benchmark data: `data/benchmarks`
- DeepSeek extraction caches: `data/llm_extraction_caches`
- Frozen predictions: `frozen_outputs`
- Paper evidence: `results/paper_evidence`
- Reproduced run outputs: `runs`
- Local model cache: `.cache/huggingface`

## Data Download

`data/` and `frozen_outputs/` are hosted on Hugging Face Dataset. After
cloning the GitHub repository, download both directories from the linked
Hugging Face Dataset and place them at the repository root.

Historical remote paths may appear only in provenance fields such as
`source_path_before_reorg` inside manifests. They are not required for reviewer
verification.
