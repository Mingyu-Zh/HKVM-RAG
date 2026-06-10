# P0 Held-Out Validation Summary

This audit uses frozen Phase1D prediction files only. It applies a deterministic hash split over example ids and reports whether saved controller gains persist on the held-out slice. It does not train controllers, run retrieval, generate answers, or call LLM APIs.

## Held-Out Metrics

| Dataset | Variant | F1 | EM | AR@10 | AR@5 |
|---|---|---:|---:|---:|---:|
| musique | colbertv2 | 56.963 | 56.300 | 49.330 | 29.088 |
| musique | phase1c_mlp_topk_4 | 55.841 | 55.183 | 54.424 | 36.416 |
| musique | controller_dense_only | 56.730 | 56.166 | 53.083 | 28.954 |
| musique | controller_hkvm_only | 55.809 | 55.139 | 58.624 | 36.416 |
| musique | controller_dense_hkvm | 64.214 | 63.807 | 64.701 | 44.951 |
| 2wiki | colbertv2 | 77.764 | 77.220 | 100.000 | 63.707 |
| 2wiki | phase1c_mlp_topk_4 | 88.505 | 88.299 | 99.059 | 92.199 |
| 2wiki | controller_dense_only | 77.641 | 77.108 | 100.000 | 63.543 |
| 2wiki | controller_hkvm_only | 88.531 | 88.325 | 100.000 | 92.225 |
| 2wiki | controller_dense_hkvm | 88.959 | 88.774 | 100.000 | 93.106 |
| hotpotqa | colbertv2 | 78.606 | 78.401 | 100.000 | 67.939 |
| hotpotqa | phase1c_mlp_topk_4 | 81.026 | 80.751 | 96.602 | 72.953 |
| hotpotqa | controller_dense_only | 78.925 | 78.656 | 100.000 | 66.727 |
| hotpotqa | controller_hkvm_only | 81.026 | 80.751 | 100.000 | 72.953 |
| hotpotqa | controller_dense_hkvm | 84.377 | 84.134 | 100.000 | 79.060 |

## Held-Out F1 Deltas

| Dataset | Reference | Delta F1 | 95% CI | p(diff <= 0) | Seed min/max |
|---|---|---:|---:|---:|---:|
| musique | colbertv2 | +7.251 | [5.423, 9.109] | 0.0000 | +6.952/+7.832 |
| musique | phase1c_mlp_topk_4 | +8.373 | [6.689, 10.138] | 0.0000 | +7.892/+8.615 |
| musique | controller_dense_only | +7.484 | [5.757, 9.249] | 0.0000 | +6.973/+8.380 |
| musique | controller_hkvm_only | +8.405 | [6.732, 10.118] | 0.0000 | +7.892/+8.708 |
| 2wiki | colbertv2 | +11.196 | [10.558, 11.837] | 0.0000 | +11.055/+11.353 |
| 2wiki | phase1c_mlp_topk_4 | +0.454 | [0.302, 0.613] | 0.0000 | +0.434/+0.484 |
| 2wiki | controller_dense_only | +11.318 | [10.669, 11.955] | 0.0000 | +11.231/+11.396 |
| 2wiki | controller_hkvm_only | +0.428 | [0.267, 0.588] | 0.0000 | +0.408/+0.458 |
| hotpotqa | colbertv2 | +5.771 | [4.909, 6.639] | 0.0000 | +5.618/+6.035 |
| hotpotqa | phase1c_mlp_topk_4 | +3.351 | [2.741, 3.958] | 0.0000 | +3.297/+3.439 |
| hotpotqa | controller_dense_only | +5.452 | [4.613, 6.325] | 0.0000 | +5.302/+5.642 |
| hotpotqa | controller_hkvm_only | +3.351 | [2.721, 3.984] | 0.0000 | +3.297/+3.439 |

## Interpretation Boundary

- Safe: deterministic held-out-slice robustness over frozen dev predictions.
- Unsafe: hidden-test, official-server, deployment-throughput, or new-training claims.
- The held-out split is an audit slice, not a substitute for an official blind test.
