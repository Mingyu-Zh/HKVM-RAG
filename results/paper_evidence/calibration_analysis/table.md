# Phase1C Paper-Facing Result Table

| Dataset | Metric | D0 weighted | Oracle D1 | Phase1B scorer | Phase1C MLP | Delta | 95% CI | p(diff <= 0) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| musique | F1 | 39.925 | 74.330 | 54.621 | 56.754 | +2.133 | [1.479, 2.787] | 0.0000 |
| musique | EM | 39.222 | 74.059 | 54.034 | 56.213 | +2.179 | [1.545, 2.786] | 0.0000 |
| musique | AR@10 | 38.643 | 73.066 | 50.310 | 53.303 | +2.993 | [2.427, 3.627] | 0.0000 |
| 2wiki | F1 | 79.299 | 90.884 | 87.212 | 88.391 | +1.179 | [1.041, 1.318] | 0.0000 |
| 2wiki | EM | 78.650 | 90.832 | 86.967 | 88.181 | +1.214 | [1.073, 1.362] | 0.0000 |
| 2wiki | AR@10 | 99.650 | 99.634 | 98.378 | 99.030 | +0.652 | [0.572, 0.732] | 0.0000 |

Notes: Phase1C MLP is `mlp_base_only_topk_4`. CI is the matched seed-example bootstrap for Phase1C MLP minus Phase1B scorer. D0 and Oracle D1 are retained as context, not deployable final baselines.
