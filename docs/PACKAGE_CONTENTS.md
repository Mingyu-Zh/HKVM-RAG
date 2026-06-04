# Package Contents

This artifact is split across two hosting platforms:

- **GitHub**: code, configs, scripts, docs, paper evidence results, and manifests.
- **Hugging Face Dataset**: `data/` (benchmarks + DeepSeek extraction caches) and
  `frozen_outputs/` (frozen prediction files).

After downloading both parts and merging them at the repository root, the
package contains:

- Core HKVM-RAG code under `hkvm_mvp/`.
- Reviewer-facing configs under `configs/`.
- Reproduction and verification entry points under `scripts/`.
- Normalized benchmark splits under `data/benchmarks/`.
- Paper-relevant DeepSeek extraction caches under `data/llm_extraction_caches/`.
- Frozen prediction outputs under `frozen_outputs/`.
- Paper-facing tables, bootstrap summaries, and claim-evidence maps under `results/paper_evidence/`.
- Compact run summaries under `results/run_summaries/`.
- SHA-256 file list and release manifest under `manifests/`.

The package intentionally excludes:

- Raw upstream benchmark archives.
- Model weights and Hugging Face caches.
- API keys, `.env` secrets, and credentials.
- Remote conda environments.
- Historical debug, smoke, dry-run, waiting-for-GPU, and temporary experiment scripts.

