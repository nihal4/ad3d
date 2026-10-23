# Offline statistics: mvtec  (5 seeds: [0, 1, 2, 3, 4], 10 categories, bootstrap B=1000, resampling unit = test object within category)

## Object-level AUROC (%), mean over categories

| channel | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std (seeds) |
|---|---|---|---|---|---|---|---|
| hyb_max | 68.88 | 68.66 | 69.75 | 66.99 | 67.04 | 68.26 | 1.21 |
| h1_hyb_max | 68.79 | 69.53 | 68.48 | 66.16 | 68.36 | 68.27 | 1.26 |

## Bootstrap 95 % CI (mean over seeds and categories)

| quantity | estimate | 95 % CI |
|---|---|---|
| hyb_max | 68.26 | [65.69, 70.92] |
| h1_hyb_max | 68.27 | [65.64, 70.87] |
| hyb_max - h1_hyb_max | -0.00 | [-1.11, +1.13] (one-sided bootstrap p = 0.5360) |

## Per-category paired difference hyb_max - h1_hyb_max (AUROC %, mean over seeds)

| category | n normal | n anomalous | hyb_max | h1_hyb_max | diff | 95 % CI |
|---|---|---|---|---|---|---|
| cookie | 28 | 103 | 56.4 | 53.8 | +2.6 | [-1.1, +5.8] |
| carrot | 27 | 132 | 70.7 | 68.2 | +2.4 | [-0.5, +5.7] |
| rope | 32 | 69 | 92.6 | 92.5 | +0.1 | [-0.0, +0.4] |
| dowel | 26 | 104 | 64.8 | 64.7 | +0.1 | [-1.4, +1.6] |
| foam | 20 | 80 | 67.1 | 67.1 | +0.0 | [+0.0, +0.0] |
| bagel | 22 | 88 | 90.4 | 90.5 | -0.1 | [-1.9, +1.6] |
| potato | 22 | 92 | 73.4 | 73.6 | -0.2 | [-1.5, +0.9] |
| tire | 25 | 87 | 54.3 | 54.8 | -0.5 | [-8.8, +7.5] |
| peach | 26 | 106 | 69.3 | 71.0 | -1.7 | [-5.2, +1.7] |
| cable_gland | 21 | 87 | 43.7 | 46.5 | -2.7 | [-7.1, +0.5] |

Categories with CI entirely above 0: 0; entirely below 0: 0; overlapping 0: 10.

## Operating points (per category, then mean over categories and seeds)

| channel | FPR on normals at 95 % TPR (%) | TPR at 5 % FPR (%) |
|---|---|---|
| hyb_max | 74.5 | 25.0 |
| h1_hyb_max | 75.4 | 26.7 |

## Defect-size strata (anomalous-point fraction tertiles over all anomalous test objects: small < 0.90 % <= medium < 2.09 % <= large)

| channel | small | medium | large |
|---|---|---|---|
| hyb_max | 65.1 | 69.5 | 66.7 |
| h1_hyb_max | 65.2 | 70.2 | 67.6 |
