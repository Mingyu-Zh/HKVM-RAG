# Adaptive Source Controller Paper-Facing Result Table

| Dataset | Variant | F1 | EM | AR@10 | Delta F1 vs ColBERTv2 | Delta F1 vs learned HKVM |
|---|---|---:|---:|---:|---:|---:|
| musique | ColBERTv2 | 58.309 | 57.758 | 49.317 | +0.000 | +1.556 |
| musique | Learned HKVM calibration | 56.754 | 56.213 | 53.303 | -1.556 | +0.000 |
| musique | Quota dense3+HKVM2 | 59.121 | 58.640 | 51.965 | +0.811 | +2.367 |
| musique | Quota dense4+HKVM1 | 58.463 | 57.978 | 51.303 | +0.154 | +1.709 |
| musique | RRF dense+HKVM | 58.420 | 57.909 | 62.943 | +0.110 | +1.666 |
| musique | RRF dense+HKVM+BM25 | 58.420 | 57.909 | 62.943 | +0.110 | +1.666 |
| musique | Controller dense-only | 58.247 | 57.758 | 52.172 | -0.062 | +1.494 |
| musique | Controller HKVM-only | 56.760 | 56.213 | 58.640 | -1.549 | +0.006 |
| musique | Controller dense+HKVM | 65.073 | 64.722 | 64.115 | +6.763 | +8.319 |
| 2wiki | ColBERTv2 | 77.763 | 77.195 | 100.000 | +0.000 | -10.629 |
| 2wiki | Learned HKVM calibration | 88.391 | 88.181 | 99.030 | +10.629 | +0.000 |
| 2wiki | Quota dense3+HKVM2 | 79.154 | 78.597 | 100.000 | +1.391 | -9.238 |
| 2wiki | Quota dense4+HKVM1 | 78.135 | 77.568 | 100.000 | +0.372 | -10.256 |
| 2wiki | RRF dense+HKVM | 86.563 | 86.230 | 100.000 | +8.800 | -1.828 |
| 2wiki | RRF dense+HKVM+BM25 | 86.563 | 86.230 | 100.000 | +8.800 | -1.828 |
| 2wiki | Controller dense-only | 77.679 | 77.107 | 100.000 | -0.084 | -10.713 |
| 2wiki | Controller HKVM-only | 88.415 | 88.205 | 100.000 | +10.652 | +0.024 |
| 2wiki | Controller dense+HKVM | 88.846 | 88.658 | 100.000 | +11.084 | +0.455 |
| hotpotqa | ColBERTv2 | 79.844 | 79.635 | 100.000 | +0.000 | -3.085 |
| hotpotqa | Learned HKVM calibration | 82.929 | 82.705 | 96.439 | +3.085 | +0.000 |
| hotpotqa | Quota dense3+HKVM2 | 80.918 | 80.711 | 100.000 | +1.074 | -2.011 |
| hotpotqa | Quota dense4+HKVM1 | 80.873 | 80.666 | 100.000 | +1.029 | -2.056 |
| hotpotqa | RRF dense+HKVM | 83.223 | 83.034 | 100.000 | +3.379 | +0.294 |
| hotpotqa | RRF dense+HKVM+BM25 | 83.223 | 83.034 | 100.000 | +3.379 | +0.294 |
| hotpotqa | Controller dense-only | 79.909 | 79.671 | 100.000 | +0.065 | -3.020 |
| hotpotqa | Controller HKVM-only | 82.929 | 82.705 | 100.000 | +3.085 | +0.000 |
| hotpotqa | Controller dense+HKVM | 85.810 | 85.627 | 100.000 | +5.966 | +2.881 |

Notes: Candidate method is `controller_dense_hkvm`. The learned HKVM comparator is `calibrated_hkvm_topk4`. Controller rows use train-to-dev controller training over saved dense and HKVM predictions with `learning_rate=0.01` and `epochs=80`.
