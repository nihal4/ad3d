# Offline statistics: mvtec  (5 seeds: [0, 1, 2, 3, 4], 10 categories, bootstrap B=1000, resampling unit = test object within category)

## Object-level AUROC (%), mean over categories

| channel | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std (seeds) |
|---|---|---|---|---|---|---|---|
| hyb_max | 68.88 | 68.66 | 69.75 | 66.99 | 67.04 | 68.26 | 1.21 |
| glob_full_p99 | 66.60 | 67.35 | 65.44 | 64.34 | 65.54 | 65.85 | 1.16 |

## Bootstrap 95 % CI (mean over seeds and categories)

| quantity | estimate | 95 % CI |
|---|---|---|
| hyb_max | 68.26 | [65.69, 70.92] |
| glob_full_p99 | 65.85 | [63.06, 68.90] |
| hyb_max - glob_full_p99 | +2.41 | [+0.29, +4.59] (one-sided bootstrap p = 0.0110) |

## Per-category paired difference hyb_max - glob_full_p99 (AUROC %, mean over seeds)

| category | n normal | n anomalous | hyb_max | glob_full_p99 | diff | 95 % CI |
|---|---|---|---|---|---|---|
| bagel | 22 | 88 | 90.4 | 82.0 | +8.4 | [+1.1, +16.2] |
| peach | 26 | 106 | 69.3 | 61.0 | +8.3 | [-1.7, +19.2] |
| dowel | 26 | 104 | 64.8 | 59.7 | +5.1 | [-0.1, +9.7] |
| carrot | 27 | 132 | 70.7 | 66.2 | +4.5 | [+0.3, +9.4] |
| potato | 22 | 92 | 73.4 | 70.0 | +3.4 | [-2.8, +9.4] |
| foam | 20 | 80 | 67.1 | 63.8 | +3.3 | [-6.5, +14.1] |
| tire | 25 | 87 | 54.3 | 53.4 | +0.9 | [-5.0, +6.6] |
| rope | 32 | 69 | 92.6 | 92.9 | -0.2 | [-3.5, +3.0] |
| cookie | 28 | 103 | 56.4 | 59.8 | -3.5 | [-8.5, +1.3] |
| cable_gland | 21 | 87 | 43.7 | 49.8 | -6.1 | [-13.6, +1.3] |

Categories with CI entirely above 0: 2; entirely below 0: 0; overlapping 0: 8.

## Operating points (per category, then mean over categories and seeds)

| channel | FPR on normals at 95 % TPR (%) | TPR at 5 % FPR (%) |
|---|---|---|
| hyb_max | 74.5 | 25.0 |
| glob_full_p99 | 77.3 | 21.0 |

## Defect-size strata (anomalous-point fraction tertiles over all anomalous test objects: small < 0.90 % <= medium < 2.09 % <= large)

| channel | small | medium | large |
|---|---|---|---|
| hyb_max | 65.1 | 69.5 | 66.7 |
| glob_full_p99 | 63.1 | 67.0 | 64.0 |
