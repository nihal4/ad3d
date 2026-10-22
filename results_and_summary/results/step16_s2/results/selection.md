# Normal-only configuration selection (no test labels used)

Development files: 16; scales: [6, 10, 20, 40]; datasets: ['mvtec', 'real3d']. Tie margin 0.5 AUROC points -> simpler option.


## 1. Descriptor base scale (rule loc10, pooling p99)

| option | mean | mvtec | real3d |
|---|---|---|---|
| k=6 | 70.41 | 66.39 | 74.42 |
| k=10 | 72.56 | 68.61 | 76.50 |
| k=20 | 76.03 | 75.56 | 76.50 |
| k=40 | 71.79 | 67.78 | 75.81 |

## 2. Matching rule (k=20, pooling p99)

| option | mean | mvtec | real3d | diff to best, 95 % CI |
|---|---|---|---|---|
| glob_cs | 70.08 | 70.83 | 69.33 | -7.12 [-11.19, -2.93] |
| glob_full | 70.91 | 72.50 | 69.33 | -6.28 [-10.47, -2.05] |
| loc10 | 76.03 | 75.56 | 76.50 | -1.17 [-3.47, +0.87] |
| loc05 | 72.78 | 68.89 | 76.68 | -4.42 [-6.74, -1.70] |
| loc15 | 75.86 | 75.28 | 76.45 | -1.34 [-4.20, +1.48] |
| knn64 | 71.50 | 66.39 | 76.62 | -5.69 [-10.05, -1.06] |
| knn256 | 75.19 | 73.06 | 77.31 | -2.01 [-6.53, +2.36] |
| soft0p5 | 73.14 | 69.72 | 76.56 | -4.06 [-8.39, -0.18] |
| soft1 | 71.00 | 65.83 | 76.16 | -6.20 [-10.83, -2.01] |
| soft2 | 69.64 | 63.06 | 76.22 | -7.56 [-12.37, -3.00] |
| cal10_k0 | 68.97 | 70.00 | 67.94 | -8.23 [-11.92, -4.47] |
| cal10 | 73.19 | 71.39 | 75.00 | -4.00 [-7.05, -0.91] |
| hyb | 77.20 | 76.39 | 78.01 | best |

## 3. Training views (rule family loc10, pooling p99)

| option | mean | mvtec | real3d | diff to best, 95 % CI |
|---|---|---|---|---|
| V0 | 74.90 | 73.06 | 76.74 | -2.29 [+nan, +nan] |
| V01 | 77.18 | nan | 77.18 | best |
| V02 | 74.61 | 73.06 | 76.16 | -2.58 [+nan, +nan] |
| M2 | 75.04 | 74.44 | 75.64 | -2.14 [+nan, +nan] |
| V012 | 76.03 | 75.56 | 76.50 | -1.15 [+nan, +nan] |

## 4. Object-score pooling (rule hyb)

| option | mean | mvtec | real3d | diff to best, 95 % CI |
|---|---|---|---|---|
| p99 | 77.20 | 76.39 | 78.01 | -1.23 [-2.59, +0.16] |
| max | 77.84 | 76.11 | 79.57 | -0.59 [-1.59, +0.36] |
| p95 | 73.70 | 79.17 | 68.23 | -4.73 [-8.74, -0.39] |
| top1pct | 78.43 | 76.94 | 79.92 | best |
| mean | 71.20 | 73.89 | 68.52 | -7.23 [-10.64, -3.77] |

## 5. Pose hypotheses

not separable for rule hyb (kept at H = 6).


## Selected (frozen) configuration

```json
{
  "max_nn": 20,
  "rule": "hyb",
  "views": "V01",
  "pooling": "top1pct",
  "mhr": true
}
```

