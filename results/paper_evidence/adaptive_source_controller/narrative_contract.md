# Phase1D Result Narrative Contract

## Use

- Describe Phase1D as a train-to-dev dense-aware evidence controller over saved ColBERTv2 and Phase1C HKVM predictions.
- The main method row is `controller_dense_hkvm` with `learning_rate=0.01` and `epochs=80`.
- Prefer the OOF-confirmed Phase1D values when writing paper-facing result tables.
- The mechanism claim is interaction-specific: dense retrieval provides semantic candidates while HKVM provides structured evidence/key signals for learned reordering.
- Report paired bootstrap against both ColBERTv2 and Phase1C HKVM.

## Avoid

- Do not imply dev labels are used for official controller training.
- Do not imply the OOF risk-resolution check proves hidden-test performance.
- Do not claim broad dense-family generality until at least one additional dense source is checked.
- Do not present the default-config run as the final Phase1D result.

## Suggested Paper Wording

We further test HKVM as a dense-aware evidence controller by training a passage-level ranker on train-side ColBERTv2 and HKVM predictions and evaluating it on dev predictions only. On MuSiQue, the dense+HKVM controller substantially improves over ColBERTv2 and Phase1C HKVM, while dense-only, HKVM-only, quota fusion, and RRF are weaker. This indicates that HKVM's benefit is strongest when used as a structured key-value signal for learned reordering over dense candidates. The train-side HKVM input is out-of-fold in this confirmation run, and the manifest marks the in-sample stacking-risk flag as false; therefore we treat the earlier train-side stacking concern as resolved for MuSiQue and 2Wiki, while preserving the train-to-dev and cached-prediction boundaries.
