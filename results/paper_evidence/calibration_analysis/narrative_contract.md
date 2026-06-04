# Phase1C Result Narrative Contract

## Use

- HKVM-RAG's Phase1C result shows that learned key/edge calibration over hypergraph-derived candidate evidence improves MuSiQue and 2Wiki train-to-dev validation.
- The conservative mainline variant is `mlp_base_only_topk_4`, using 20 base direct features and a nonlinear MLP calibration layer.
- Structural/proxy features are informative diagnostics and support-recall signals, but are not the current source of the best QA F1.

## Avoid

- Do not say full structural proxies produce the gain.
- Do not present this as hidden-test performance.
- Do not make complement selection the main method unless a later ordering/usefulness objective validates it.

## Suggested Paper Wording

Phase1C replaces the linear support scorer with a learned calibration layer over deployable edge-key features. Across MuSiQue and 2Wiki, `mlp_base_only_topk_4` consistently improves F1/EM/AR@10 over the Phase1B scorer under three training seeds, with matched bootstrap confidence intervals fully above zero. The ablation shows that current structural proxy features primarily increase support recall rather than answer-level F1, motivating future generator-aware ordering objectives.
