# Offline statistics: real3d  (5 seeds: [0, 1, 2, 3, 4], 12 categories, bootstrap B=1000, resampling unit = test object within category)

## Object-level AUROC (%), mean over categories

| channel | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std (seeds) |
|---|---|---|---|---|---|---|---|
| hyb_max | 85.22 | 87.05 | 87.48 | 86.60 | 86.21 | 86.51 | 0.86 |
| glob_full_max | 74.97 | 74.99 | 76.34 | 75.10 | 75.20 | 75.32 | 0.58 |

## Bootstrap 95 % CI (mean over seeds and categories)

| quantity | estimate | 95 % CI |
|---|---|---|
| hyb_max | 86.51 | [84.61, 88.26] |
| glob_full_max | 75.32 | [72.91, 77.67] |
| hyb_max - glob_full_max | +11.19 | [+9.15, +13.26] (one-sided bootstrap p = 0.0000) |

## Per-category paired difference hyb_max - glob_full_max (AUROC %, mean over seeds)

| category | n normal | n anomalous | hyb_max | glob_full_max | diff | 95 % CI |
|---|---|---|---|---|---|---|
| seahorse | 50 | 50 | 80.4 | 57.6 | +22.8 | [+14.6, +30.9] |
| car | 50 | 50 | 89.4 | 67.7 | +21.8 | [+12.0, +31.4] |
| fish | 50 | 50 | 99.9 | 84.9 | +15.0 | [+8.5, +23.5] |
| airplane | 50 | 50 | 84.0 | 69.6 | +14.4 | [+5.3, +22.9] |
| duck | 50 | 50 | 88.5 | 74.7 | +13.8 | [+6.6, +21.6] |
| starfish | 50 | 50 | 82.3 | 68.6 | +13.7 | [+7.7, +19.6] |
| toffees | 50 | 50 | 86.5 | 73.5 | +13.0 | [+6.2, +20.6] |
| gemstone | 50 | 50 | 88.6 | 79.7 | +8.9 | [+1.8, +16.4] |
| chicken | 52 | 54 | 77.6 | 70.1 | +7.5 | [+0.2, +15.0] |
| diamond | 50 | 50 | 100.0 | 94.7 | +5.3 | [+2.3, +9.0] |
| candybar | 50 | 50 | 99.0 | 96.3 | +2.7 | [-0.5, +7.1] |
| shell | 52 | 48 | 61.8 | 66.4 | -4.5 | [-11.9, +3.3] |

Categories with CI entirely above 0: 10; entirely below 0: 0; overlapping 0: 2.

## Operating points (per category, then mean over categories and seeds)

| channel | FPR on normals at 95 % TPR (%) | TPR at 5 % FPR (%) |
|---|---|---|
| hyb_max | 39.9 | 49.2 |
| glob_full_max | 67.6 | 29.7 |

## Defect-size strata (anomalous-point fraction tertiles over all anomalous test objects: small < 1.32 % <= medium < 2.90 % <= large)

| channel | small | medium | large |
|---|---|---|---|
| hyb_max | 82.4 | 85.5 | 93.1 |
| glob_full_max | 64.9 | 77.6 | 86.5 |
