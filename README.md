# HKVM-RAG: Key-Value-Separated Hypergraph Evidence Organization for Multi-Hop RAG

[![Paper](https://img.shields.io/badge/paper-under_review-orange)](https://github.com/Mingyu-Zh/HKVM-RAG)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![Artifact](https://img.shields.io/badge/artifact-available-brightgreen)](https://github.com/Mingyu-Zh/HKVM-RAG)

**Mingyu Zhang, Ying Ma\***  
*Faculty of Computing, Harbin Institute of Technology*  
\*Corresponding author

---

## Abstract

Multi-hop RAG retrieval is often evaluated as passage matching, yet answer generation depends on how evidence is organized under fixed budgets. We investigate this layer by separating key-side retrieval structures from value-side passage text. HKVM-RAG indexes LLM-extracted evidence tuples with answer-path hyperedges as retrieval keys while preserving passages as answer values. Under a fixed evidence substrate, answer-path hypergraph keys improve over pairwise KG-PPR on 2WikiMultiHopQA and MuSiQue, but HotpotQA shows that better structured support coverage need not improve standalone answer F1. A dense-aware controller combines frozen ColBERTv2 and HKVM features, reaching 88.846 F1 on 2WikiMultiHopQA (+11.084), 65.073 F1 on MuSiQue (+6.763), and 85.810 F1 on HotpotQA (+5.966). Source-level ablations show that matched alternative structured sources do not match the WHG-KV gains. These findings provide bounded evidence that key-value-separated hypergraph organization is a reusable mechanism for evidence control in multi-hop RAG.

---

## Conceptual Overview

<p align="center">
  <img src="Fig_1.png" alt="HKVM-RAG Concept" width="85%">
</p>

**Left:** Dense retrieval scores passages independently. **Center:** Pairwise KG retrieval uses binary entity-relation keys. **Right:** HKVM-RAG uses an answer-path hyperedge as one higher-order retrieval key that maps back to multiple passage values.

---

## Key Results

| Dataset | WHG-KV vs KG-PPR (ΔF1) | Dense-aware Controller (F1 / Δ vs ColBERTv2) |
|---------|:----------------------:|:--------------------------------------------:|
| 2WikiMultiHopQA | **+3.426** [2.877, 3.984] | **88.846** (+11.084) |
| MuSiQue | **+3.592** [1.917, 5.155] | **65.073** (+6.763) |
| HotpotQA | −0.689 [−1.453, +0.071] | **85.810** (+5.966) |

> All controller gains have paired-bootstrap 95% CIs fully above zero. WHG-KV is the strongest structured complement by point estimate across all nine dataset×first-stage rows. See the [supplementary material](supplemental_material.pdf) for full bootstrap tables.

---

## Method

<p align="center">
  <img src="Fig_2.png" alt="HKVM-RAG Architecture" width="100%">
</p>

**(a)** A frozen LLM evidence tuple cache stores passage-level relation triples with confidence fields. **(b)** HKVM assembles answer-path hyperedge keys, weights them with extractor confidence, diffuses query-seeded scores, and projects to passage-value scores. **(c)** The dense-aware controller aligns frozen ColBERTv2 and HKVM predictions by passage id and trains a logistic ranker to reorder the evidence context.

---

## Quick Verification

```bash
git clone https://github.com/Mingyu-Zh/HKVM-RAG.git
cd HKVM-RAG
# Download data/ and frozen_outputs/ from Hugging Face:
# https://huggingface.co/datasets/MingY-Zh/HKVM-RAG

python scripts/verify_paper_evidence.py
```

**Expected output:** `checks=63 bad=0`

Then verify file integrity:

```bash
shasum -a 256 -c manifests/FILES.sha256
```

---

## Fixed-Substrate Rerun

```bash
conda env create -f environment.yml
conda activate hkvm

python scripts/reproduce_fixed_substrate.py \
  --setting 2wiki \
  --methods bm25,kg_ppr,weighted_hg_kv \
  --runs 1 \
  --limit 20 \
  --output_dir runs/smoke_2wiki
```

Full reruns: `--setting 2wiki`, `musique`, or `hotpotqa`.

---

## Repository Structure

| Path | Purpose |
|---|---|
| `hkvm_mvp/` | Core HKVM-RAG implementation |
| `configs/` | Reviewer-facing configs named by paper concept |
| `scripts/reproduce_fixed_substrate.py` | Entry point for fixed-substrate reruns |
| `scripts/verify_paper_evidence.py` | Deterministic verification of paper-facing results |
| `scripts/build_release_manifest.py` | Generate SHA-256 file manifest |
| `docs/` | Release, attribution, and reproduction notes |
| `results/paper_evidence/` | Tables, bootstrap summaries, claim-evidence maps |
| `supplemental_material.pdf` | Supplementary material (4 pages) |
| `manifests/FILES.sha256` | SHA-256 manifest for integrity verification |

---

## Hosting

| Platform | Contents |
|---|---|
| **GitHub** | Code, configs, scripts, docs, results, manifests, supplement |
| [**Hugging Face**](https://huggingface.co/datasets/MingY-Zh/HKVM-RAG) | `data/` (benchmarks + DeepSeek extraction caches) and `frozen_outputs/` (frozen predictions) |

---

## Citation

If you use this code or data in your research, please cite the accompanying paper:

```bibtex
@misc{zhang2025hkvmrag,
  title        = {{HKVM-RAG}: Key-Value-Separated Hypergraph Evidence Organization for Multi-Hop {RAG}},
  author       = {Zhang, Mingyu and Ma, Ying},
  year         = {2025},
  howpublished = {\url{https://github.com/Mingyu-Zh/HKVM-RAG}},
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
