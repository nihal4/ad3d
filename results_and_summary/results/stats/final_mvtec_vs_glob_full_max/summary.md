# Offline statistics: mvtec  (5 seeds: [0, 1, 2, 3, 4], 10 categories, bootstrap B=1000, resampling unit = test object within category)

## Object-level AUROC (%), mean over categories

| channel | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std (seeds) |
|---|---|---|---|---|---|---|---|
| hyb_max | 68.88 | 68.66 | 69.75 | 66.99 | 67.04 | 68.26 | 1.21 |
| glob_full_max | 65.47 | 66.87 | 63.59 | 63.36 | 64.67 | 64.79 | 1.44 |

## Bootstrap 95 % CI (mean over seeds and categories)

| quantity | estimate | 95 % CI |
|---|---|---|
| hyb_max | 68.26 | [65.69, 70.92] |
| glob_full_max | 64.79 | [62.07, 67.65] |
| hyb_max - glob_full_max | +3.47 | [+1.53, +5.59] (one-sided bootstrap p = 0.0000) |

## Per-category paired difference hyb_max - glob_full_max (AUROC %, mean over seeds)

| category | n normal | n anomalous | hyb_max | glob_full_max | diff | 95 % CI |
|---|---|---|---|---|---|---|
| peach | 26 | 106 | 69.3 | 57.2 | +12.1 | [+1.2, +24.2] |
| dowel | 26 | 104 | 64.8 | 53.8 | +11.0 | [+5.6, +16.1] |
| bagel | 22 | 88 | 90.4 | 83.9 | +6.5 | [-0.2, +13.3] |
| foam | 20 | 80 | 67.1 | 62.2 | +4.9 | [-3.1, +12.7] |
| carrot | 27 | 132 | 70.7 | 66.0 | +4.7 | [+1.2, +8.4] |
| potato | 22 | 92 | 73.4 | 70.6 | +2.8 | [-3.6, +8.8] |
| tire | 25 | 87 | 54.3 | 52.0 | +2.3 | [-2.3, +7.0] |
| rope | 32 | 69 | 92.6 | 92.2 | +0.5 | [-2.9, +3.8] |
| cookie | 28 | 103 | 56.4 | 59.9 | -3.5 | [-8.6, +1.2] |
| cable_gland | 21 | 87 | 43.7 | 50.2 | -6.5 | [-14.1, +0.6] |

Categories with CI entirely above 0: 3; entirely below 0: 0; overlapping 0: 7.

## Operating points (per category, then mean over categories and seeds)

| channel | FPR on normals at 95 % TPR (%) | TPR at 5 % FPR (%) |
|---|---|---|
| hyb_max | 74.5 | 25.0 |
| glob_full_max | 80.3 | 20.6 |

## Defect-size strata (anomalous-point fraction tertiles over all anomalous test objects: small < 0.90 % <= medium < 2.09 % <= large)

| channel | small | medium | large |
|---|---|---|---|
| hyb_max | 65.1 | 69.5 | 66.7 |
| glob_full_max | 63.0 | 65.6 | 62.0 |
