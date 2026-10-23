# Offline statistics: real3d  (5 seeds: [0, 1, 2, 3, 4], 12 categories, bootstrap B=1000, resampling unit = test object within category)

## Object-level AUROC (%), mean over categories

| channel | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std (seeds) |
|---|---|---|---|---|---|---|---|
| hyb_max | 85.22 | 87.05 | 87.48 | 86.60 | 86.21 | 86.51 | 0.86 |
| h1_hyb_max | 85.35 | 87.08 | 87.37 | 86.73 | 86.23 | 86.55 | 0.79 |

## Bootstrap 95 % CI (mean over seeds and categories)

| quantity | estimate | 95 % CI |
|---|---|---|
| hyb_max | 86.51 | [84.61, 88.26] |
| h1_hyb_max | 86.55 | [84.64, 88.31] |
| hyb_max - h1_hyb_max | -0.04 | [-0.26, +0.20] (one-sided bootstrap p = 0.3810) |

## Per-category paired difference hyb_max - h1_hyb_max (AUROC %, mean over seeds)

| category | n normal | n anomalous | hyb_max | h1_hyb_max | diff | 95 % CI |
|---|---|---|---|---|---|---|
| starfish | 50 | 50 | 82.3 | 81.9 | +0.5 | [-1.0, +2.0] |
| airplane | 50 | 50 | 84.0 | 83.7 | +0.3 | [-0.2, +1.0] |
| seahorse | 50 | 50 | 80.4 | 80.2 | +0.3 | [+0.0, +0.6] |
| gemstone | 50 | 50 | 88.6 | 88.5 | +0.1 | [-0.5, +0.8] |
| candybar | 50 | 50 | 99.0 | 98.9 | +0.0 | [-0.3, +0.3] |
| diamond | 50 | 50 | 100.0 | 100.0 | +0.0 | [+0.0, +0.0] |
| fish | 50 | 50 | 99.9 | 99.9 | +0.0 | [+0.0, +0.0] |
| car | 50 | 50 | 89.4 | 89.5 | -0.0 | [-0.2, +0.2] |
| toffees | 50 | 50 | 86.5 | 86.6 | -0.0 | [-0.2, +0.2] |
| duck | 50 | 50 | 88.5 | 88.7 | -0.2 | [-0.6, +0.2] |
| shell | 52 | 48 | 61.8 | 62.0 | -0.2 | [-2.0, +1.4] |
| chicken | 52 | 54 | 77.6 | 78.8 | -1.2 | [-2.4, -0.3] |

Categories with CI entirely above 0: 1; entirely below 0: 1; overlapping 0: 10.

## Operating points (per category, then mean over categories and seeds)

| channel | FPR on normals at 95 % TPR (%) | TPR at 5 % FPR (%) |
|---|---|---|
| hyb_max | 39.9 | 49.2 |
| h1_hyb_max | 39.7 | 49.4 |

## Defect-size strata (anomalous-point fraction tertiles over all anomalous test objects: small < 1.32 % <= medium < 2.90 % <= large)

| channel | small | medium | large |
|---|---|---|---|
| hyb_max | 82.4 | 85.5 | 93.1 |
| h1_hyb_max | 82.3 | 85.4 | 93.3 |
