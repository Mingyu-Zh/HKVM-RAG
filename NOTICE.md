# HKVM-RAG Artifact Notice

This notice covers the HKVM-RAG ICDE 2027 artifact package. The code developed
for HKVM-RAG is released under the MIT License. Benchmark datasets, model
weights, and provider services remain governed by their own upstream terms.

## HKVM-RAG Code And Generated Experiment Artifacts

- HKVM-RAG code, configs, scripts, documentation, table exports, manifests, and
  generated experiment outputs in this package are released by the HKVM-RAG
  authors under the MIT License unless a file explicitly states otherwise.
- The package includes normalized processed benchmark files, deterministic
  train subsets, frozen prediction files, metric summaries, bootstrap outputs,
  and LLM extraction caches used for reproducibility.

## Upstream Benchmark Datasets

The original benchmark datasets are public and are not relicensed by this
artifact package.

| Dataset | Upstream source | Upstream license / terms |
|---|---|---|
| 2WikiMultiHopQA | `https://github.com/Alab-NII/2wikimultihop` | Apache-2.0 according to the upstream repository. |
| MuSiQue | `https://github.com/stonybrooknlp/musique` | CC BY 4.0 according to the upstream repository. |
| HotpotQA | `https://hotpotqa.github.io/` | CC BY-SA 4.0 according to the project site. |

The files under `data/processed/` and `data/train_subsets/` are normalized or
subsetted derivatives created for the HKVM-RAG experiments. Users should cite
and comply with the upstream dataset licenses when using those files.

## AI-Generated Extraction Outputs

The files under `data/llm_caches/` contain AI-generated extraction outputs
produced by DeepSeek V4 Flash from public benchmark passages. These cached
triples and sidecar manifests are reproducibility artifacts for the HKVM-RAG
experiments, not new benchmark labels or human annotations.

When publishing, redistributing, or reusing these cached outputs, users should
preserve the indication that they are AI-generated extraction outputs and
comply with both upstream dataset licenses and provider terms.

## Models And External Services

This package does not include Contriever, ColBERTv2, HuggingFace model caches,
DeepSeek service credentials, or any API keys. Reproducing dense retrieval or
regenerating LLM caches requires obtaining the relevant models or service access
from their respective providers.

## Excluded Materials

This package intentionally excludes:

- raw upstream benchmark archives;
- model weights and HuggingFace caches;
- API keys, `.env` secrets, and credentials;
- remote conda environments and local machine state;
- smoke, dry-run, corrupted, and temporary debug outputs.

