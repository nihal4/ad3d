# Offline statistics: mvtec  (5 seeds: [0, 1, 2, 3, 4], 10 categories, bootstrap B=1000, resampling unit = test object within category)

## Object-level AUROC (%), mean over categories

| channel | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std (seeds) |
|---|---|---|---|---|---|---|---|
| hyb_max | 68.88 | 68.66 | 69.75 | 66.99 | 67.04 | 68.26 | 1.21 |
| loc10_max | 64.34 | 60.73 | 66.79 | 62.81 | 61.77 | 63.29 | 2.37 |

## Bootstrap 95 % CI (mean over seeds and categories)

| quantity | estimate | 95 % CI |
|---|---|---|
| hyb_max | 68.26 | [65.69, 70.92] |
| loc10_max | 63.29 | [60.78, 66.04] |
| hyb_max - loc10_max | +4.98 | [+2.59, +7.27] (one-sided bootstrap p = 0.0000) |

## Per-category paired difference hyb_max - loc10_max (AUROC %, mean over seeds)

| category | n normal | n anomalous | hyb_max | loc10_max | diff | 95 % CI |
|---|---|---|---|---|---|---|
| bagel | 22 | 88 | 90.4 | 66.7 | +23.6 | [+18.1, +29.6] |
| cookie | 28 | 103 | 56.4 | 46.5 | +9.9 | [+1.2, +18.6] |
| carrot | 27 | 132 | 70.7 | 62.2 | +8.5 | [+1.3, +16.5] |
| peach | 26 | 106 | 69.3 | 63.6 | +5.7 | [-2.1, +12.9] |
| foam | 20 | 80 | 67.1 | 62.3 | +4.8 | [+0.0, +9.9] |
| potato | 22 | 92 | 73.4 | 68.6 | +4.8 | [-6.9, +16.9] |
| tire | 25 | 87 | 54.3 | 54.7 | -0.3 | [-3.9, +3.1] |
| cable_gland | 21 | 87 | 43.7 | 44.6 | -0.9 | [-9.0, +8.9] |
| rope | 32 | 69 | 92.6 | 95.3 | -2.7 | [-4.6, -1.1] |
| dowel | 26 | 104 | 64.8 | 68.3 | -3.6 | [-7.8, +0.8] |

Categories with CI entirely above 0: 4; entirely below 0: 1; overlapping 0: 5.

## Operating points (per category, then mean over categories and seeds)

| channel | FPR on normals at 95 % TPR (%) | TPR at 5 % FPR (%) |
|---|---|---|
| hyb_max | 74.5 | 25.0 |
| loc10_max | 79.3 | 21.5 |

## Defect-size strata (anomalous-point fraction tertiles over all anomalous test objects: small < 0.90 % <= medium < 2.09 % <= large)

| channel | small | medium | large |
|---|---|---|---|
| hyb_max | 65.1 | 69.5 | 66.7 |
| loc10_max | 61.9 | 63.5 | 63.3 |
