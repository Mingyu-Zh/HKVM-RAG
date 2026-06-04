# Results Layout

This directory is organized by paper-facing experiment semantics rather than historical engineering phases.

- `paper_evidence/`: compact tables, bootstrap outputs, claim-evidence maps, and narrative contracts used by the manuscript.
- `run_summaries/`: lightweight run manifests, summaries, per-seed metrics, fixed-substrate baselines, and controller model metadata.
- `indexes/result_file_index.json`: provenance map from each release path back to the original staged path.

Large frozen prediction JSONL files are stored outside this directory under `frozen_outputs/` so they can be uploaded as data artifacts, for example to Hugging Face.
