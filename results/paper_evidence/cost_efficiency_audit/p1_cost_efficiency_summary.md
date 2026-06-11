# P1 Cost/Efficiency Audit

This audit is read-only. It replays saved controller models over saved Phase1D prediction files and summarizes fixed-substrate timing/resource records. It does not train, retrieve, generate answers, or call LLM extraction APIs.

## Controller Replay Timing

| Dataset | Seeds | Examples/seed | Candidates/seed | ms/example | us/candidate | examples/sec | candidates/sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wiki | 3 | 12576.0 | 125237.0 | 0.2072 | 20.8100 | 4868.8 | 48485.4 |
| hotpotqa | 3 | 7405.0 | 73700.0 | 0.2001 | 20.1027 | 5110.0 | 50858.3 |
| musique | 3 | 2417.0 | 33939.7 | 0.2882 | 20.5222 | 3545.8 | 49789.9 |
| all_seed_rows | 9 | 7466.0 | 77625.6 | 0.2318 | 20.4783 | 4508.2 | 49711.2 |

## Fixed-Substrate Retrieval Scoring Latency

| Dataset | Method | ms/example | Avg retrieval tokens |
|---|---|---:|---:|
| 2wiki | bm25 | 0.2484 | 424.5 |
| 2wiki | colbertv2 | 29.7707 | 382.3 |
| 2wiki | contriever | 19.1964 | 303.3 |
| 2wiki | kg_ppr | 0.5133 | 392.5 |
| 2wiki | weighted_hg_kv | 0.1571 | 393.9 |
| hotpotqa | bm25 | 0.4822 | 484.0 |
| hotpotqa | colbertv2 | 61.6888 | 438.2 |
| hotpotqa | contriever | 26.9177 | 400.7 |
| hotpotqa | kg_ppr | 0.7343 | 439.8 |
| hotpotqa | weighted_hg_kv | 0.2639 | 432.9 |
| musique | bm25 | 0.5899 | 475.0 |
| musique | colbertv2 | 53.1897 | 460.1 |
| musique | contriever | 36.5927 | 416.0 |
| musique | kg_ppr | 1.2162 | 426.5 |
| musique | weighted_hg_kv | 0.3299 | 411.1 |

## Extraction Scope

- Main total: 202,176 cached passage-level extractor inputs.
- Train total: 399,491 cached passage-level extractor inputs.

## Artifact Resources

- Audited artifact release manifest: `/home/zzz/hkvm_runs/p1_cost_efficiency_inputs/artifact_release_manifest.json`.
- Total audited files: 498; total bytes: 2,291,468,263.
- other: 28 files, 144,699 bytes.
- benchmark_data: 9 files, 355,293,085 bytes.
- llm_cache: 13 files, 734,269,603 bytes.
- data_documentation: 1 files, 253 bytes.
- documentation: 4 files, 4,987 bytes.
- frozen_output: 143 files, 1,200,203,334 bytes.
- result: 296 files, 1,538,964 bytes.
- packaging_script: 4 files, 13,338 bytes.

## Claim Boundary

- Safe: CPU-side controller replay overhead over frozen prediction files.
- Safe: fixed-substrate retrieval scoring latency already recorded in run summaries.
- Safe: cache/resource footprint for reproducing the submitted evidence matrix.
- Unsafe: deployment throughput, online RAG serving latency, full-corpus indexing throughput, or fresh LLM extraction cost.
