# Offline statistics: real3d  (5 seeds: [0, 1, 2, 3, 4], 12 categories, bootstrap B=1000, resampling unit = test object within category)

## Object-level AUROC (%), mean over categories

| channel | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std (seeds) |
|---|---|---|---|---|---|---|---|
| hyb_max | 85.22 | 87.05 | 87.48 | 86.60 | 86.21 | 86.51 | 0.86 |
| glob_full_p99 | 78.16 | 77.73 | 79.10 | 78.32 | 78.29 | 78.32 | 0.50 |

## Bootstrap 95 % CI (mean over seeds and categories)

| quantity | estimate | 95 % CI |
|---|---|---|
| hyb_max | 86.51 | [84.61, 88.26] |
| glob_full_p99 | 78.32 | [75.88, 80.75] |
| hyb_max - glob_full_p99 | +8.19 | [+6.23, +10.24] (one-sided bootstrap p = 0.0000) |

## Per-category paired difference hyb_max - glob_full_p99 (AUROC %, mean over seeds)

| category | n normal | n anomalous | hyb_max | glob_full_p99 | diff | 95 % CI |
|---|---|---|---|---|---|---|
| car | 50 | 50 | 89.4 | 65.6 | +23.9 | [+14.2, +33.7] |
| fish | 50 | 50 | 99.9 | 82.7 | +17.2 | [+9.7, +26.8] |
| seahorse | 50 | 50 | 80.4 | 63.4 | +17.0 | [+9.4, +25.0] |
| airplane | 50 | 50 | 84.0 | 69.8 | +14.3 | [+6.2, +22.2] |
| duck | 50 | 50 | 88.5 | 74.6 | +13.9 | [+6.2, +22.5] |
| toffees | 50 | 50 | 86.5 | 79.0 | +7.6 | [+0.5, +15.4] |
| gemstone | 50 | 50 | 88.6 | 85.0 | +3.5 | [-3.1, +10.6] |
| chicken | 52 | 54 | 77.6 | 74.6 | +3.1 | [-3.2, +9.7] |
| candybar | 50 | 50 | 99.0 | 96.2 | +2.8 | [-0.1, +6.8] |
| diamond | 50 | 50 | 100.0 | 98.8 | +1.2 | [+0.2, +2.8] |
| starfish | 50 | 50 | 82.3 | 85.0 | -2.7 | [-6.6, +1.4] |
| shell | 52 | 48 | 61.8 | 65.2 | -3.4 | [-8.9, +2.6] |

Categories with CI entirely above 0: 7; entirely below 0: 0; overlapping 0: 5.

## Operating points (per category, then mean over categories and seeds)

| channel | FPR on normals at 95 % TPR (%) | TPR at 5 % FPR (%) |
|---|---|---|
| hyb_max | 39.9 | 49.2 |
| glob_full_p99 | 61.5 | 33.6 |

## Defect-size strata (anomalous-point fraction tertiles over all anomalous test objects: small < 1.32 % <= medium < 2.90 % <= large)

| channel | small | medium | large |
|---|---|---|---|
| hyb_max | 82.4 | 85.5 | 93.1 |
| glob_full_p99 | 68.2 | 80.8 | 90.2 |
