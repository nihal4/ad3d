# Offline statistics: real3d  (3 seeds: [0, 1, 2], 12 categories, bootstrap B=2000, resampling unit = test object within category)

## Object-level AUROC (%), mean over categories

| channel | seed 0 | seed 1 | seed 2 | mean | std (seeds) |
|---|---|---|---|---|---|
| loc10_p99 | 85.09 | 86.46 | 87.93 | 86.49 | 1.42 |
| glob_p99 | 81.55 | 80.68 | 82.15 | 81.46 | 0.74 |
| loc10_max | 85.49 | 87.58 | 89.11 | 87.39 | 1.82 |

## Bootstrap 95 % CI (mean over seeds and categories)

| quantity | estimate | 95 % CI |
|---|---|---|
| loc10_p99 | 86.49 | [84.74, 88.21] |
| glob_p99 | 81.46 | [79.16, 83.76] |
| loc10_max | 87.39 | [85.71, 89.02] |
| loc10_p99 - glob_p99 | +5.03 | [+3.03, +7.03] (one-sided bootstrap p = 0.0000) |

## Per-category paired difference loc10_p99 - glob_p99 (AUROC %, mean over seeds)

| category | n normal | n anomalous | loc10_p99 | glob_p99 | diff | 95 % CI |
|---|---|---|---|---|---|---|
| fish | 50 | 50 | 99.9 | 84.1 | +15.8 | [+8.7, +24.5] |
| car | 50 | 50 | 84.1 | 68.8 | +15.3 | [+8.3, +23.1] |
| duck | 50 | 50 | 90.1 | 75.1 | +15.0 | [+7.1, +23.7] |
| airplane | 50 | 50 | 82.1 | 74.4 | +7.7 | [+1.1, +15.0] |
| seahorse | 50 | 50 | 89.0 | 82.1 | +6.9 | [-0.0, +14.0] |
| toffees | 50 | 50 | 92.0 | 85.8 | +6.2 | [+1.3, +12.1] |
| candybar | 50 | 50 | 100.0 | 98.4 | +1.6 | [+0.3, +3.7] |
| diamond | 50 | 50 | 100.0 | 98.6 | +1.4 | [+0.2, +3.3] |
| starfish | 50 | 50 | 91.1 | 90.0 | +1.2 | [-1.9, +4.2] |
| chicken | 52 | 54 | 76.6 | 77.4 | -0.8 | [-7.7, +6.4] |
| gemstone | 50 | 50 | 78.1 | 82.0 | -4.0 | [-14.1, +6.7] |
| shell | 52 | 48 | 54.9 | 60.9 | -6.0 | [-14.4, +2.2] |

Categories with CI entirely above 0: 7; entirely below 0: 0; overlapping 0: 5.

## Operating points (per category, then mean over categories and seeds)

| channel | FPR on normals at 95 % TPR (%) | TPR at 5 % FPR (%) |
|---|---|---|
| loc10_p99 | 34.1 | 51.2 |
| glob_p99 | 59.4 | 42.8 |
| loc10_max | 32.2 | 53.9 |

## Defect-size strata (anomalous-point fraction tertiles over all anomalous test objects: small < 1.31 % <= medium < 2.92 % <= large)

| channel | small | medium | large |
|---|---|---|---|
| loc10_p99 | 83.1 | 87.2 | 91.6 |
| glob_p99 | 72.3 | 82.9 | 93.3 |
| loc10_max | 85.0 | 86.8 | 91.5 |
