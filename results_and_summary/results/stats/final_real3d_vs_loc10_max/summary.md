# Offline statistics: real3d  (5 seeds: [0, 1, 2, 3, 4], 12 categories, bootstrap B=1000, resampling unit = test object within category)

## Object-level AUROC (%), mean over categories

| channel | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std (seeds) |
|---|---|---|---|---|---|---|---|
| hyb_max | 85.22 | 87.05 | 87.48 | 86.60 | 86.21 | 86.51 | 0.86 |
| loc10_max | 84.60 | 86.36 | 87.25 | 85.39 | 85.27 | 85.77 | 1.04 |

## Bootstrap 95 % CI (mean over seeds and categories)

| quantity | estimate | 95 % CI |
|---|---|---|
| hyb_max | 86.51 | [84.61, 88.26] |
| loc10_max | 85.77 | [84.04, 87.58] |
| hyb_max - loc10_max | +0.74 | [-0.30, +1.71] (one-sided bootstrap p = 0.0780) |

## Per-category paired difference hyb_max - loc10_max (AUROC %, mean over seeds)

| category | n normal | n anomalous | hyb_max | loc10_max | diff | 95 % CI |
|---|---|---|---|---|---|---|
| car | 50 | 50 | 89.4 | 83.3 | +6.2 | [+0.9, +12.1] |
| toffees | 50 | 50 | 86.5 | 83.4 | +3.1 | [-0.9, +7.3] |
| shell | 52 | 48 | 61.8 | 59.3 | +2.5 | [-0.5, +5.7] |
| gemstone | 50 | 50 | 88.6 | 87.4 | +1.2 | [-1.7, +4.0] |
| airplane | 50 | 50 | 84.0 | 83.5 | +0.5 | [-3.9, +4.9] |
| chicken | 52 | 54 | 77.6 | 77.2 | +0.4 | [-3.4, +3.9] |
| diamond | 50 | 50 | 100.0 | 100.0 | +0.0 | [+0.0, +0.0] |
| fish | 50 | 50 | 99.9 | 100.0 | -0.0 | [-0.2, +0.0] |
| candybar | 50 | 50 | 99.0 | 99.6 | -0.7 | [-1.9, +0.3] |
| duck | 50 | 50 | 88.5 | 89.2 | -0.7 | [-3.3, +1.5] |
| starfish | 50 | 50 | 82.3 | 84.0 | -1.6 | [-4.3, +1.3] |
| seahorse | 50 | 50 | 80.4 | 82.5 | -2.0 | [-7.4, +2.6] |

Categories with CI entirely above 0: 1; entirely below 0: 0; overlapping 0: 11.

## Operating points (per category, then mean over categories and seeds)

| channel | FPR on normals at 95 % TPR (%) | TPR at 5 % FPR (%) |
|---|---|---|
| hyb_max | 39.9 | 49.2 |
| loc10_max | 37.1 | 50.6 |

## Defect-size strata (anomalous-point fraction tertiles over all anomalous test objects: small < 1.32 % <= medium < 2.90 % <= large)

| channel | small | medium | large |
|---|---|---|---|
| hyb_max | 82.4 | 85.5 | 93.1 |
| loc10_max | 82.3 | 85.0 | 91.7 |
