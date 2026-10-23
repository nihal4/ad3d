# Normal-only configuration selection (no test labels used)

Development files: 16; scales: [6, 10, 20, 40]; datasets: ['mvtec', 'shapenet']. Tie margin 0.5 AUROC points -> simpler option.


## 1. Descriptor base scale (rule loc10, pooling p99)

| option | mean | mvtec | shapenet |
|---|---|---|---|
| k=6 | 68.45 | 65.73 | 71.18 |
| k=10 | 68.46 | 65.14 | 71.79 |
| k=20 | 70.05 | 67.57 | 72.53 |
| k=40 | 69.15 | 66.53 | 71.77 |

## 2. Matching rule (k=20, pooling p99)

| option | mean | mvtec | shapenet | diff to best, 95 % CI |
|---|---|---|---|---|
| glob_cs | 71.98 | 70.08 | 73.89 | -2.19 [-4.69, +0.20] |
| glob_full | 72.04 | 70.64 | 73.44 | -2.14 [-4.61, +0.30] |
| loc10 | 70.05 | 67.57 | 72.53 | -4.12 [-6.49, -2.19] |
| loc05 | 68.55 | 64.79 | 72.31 | -5.62 [-8.95, -2.74] |
| loc15 | 70.72 | 67.37 | 74.08 | -3.45 [-6.25, -1.25] |
| knn64 | 67.79 | 63.78 | 71.79 | -6.39 [-10.24, -3.21] |
| knn256 | 68.91 | 65.38 | 72.43 | -5.27 [-9.70, -1.90] |
| soft0p5 | 69.15 | 65.62 | 72.67 | -5.03 [-8.65, -2.24] |
| soft1 | 68.72 | 65.03 | 72.40 | -5.46 [-9.01, -2.71] |
| soft2 | 68.02 | 64.76 | 71.28 | -6.15 [-9.79, -3.12] |
| cal10_k0 | 66.58 | 65.50 | 67.66 | -7.60 [-10.13, -5.23] |
| cal10 | 68.69 | 65.64 | 71.74 | -5.49 [-7.80, -3.31] |
| hyb | 74.18 | 72.10 | 76.25 | best |

## 3. Training views (rule family loc10, pooling p99)

| option | mean | mvtec | shapenet | diff to best, 95 % CI |
|---|---|---|---|---|
| V0 | 69.06 | 67.66 | 70.47 | -0.99 [-1.70, -0.24] |
| V01 | 70.05 | 67.57 | 72.53 | best |
| V02 | 69.06 | 67.66 | 70.47 | -0.99 [-1.70, -0.24] |
| M2 | 69.57 | 67.81 | 71.32 | -0.49 [-1.14, +0.19] |
| V012 | 70.05 | 67.57 | 72.53 | +0.00 [+0.00, +0.00] |

## 4. Object-score pooling (rule hyb)

| option | mean | mvtec | shapenet | diff to best, 95 % CI |
|---|---|---|---|---|
| p99 | 74.18 | 72.10 | 76.25 | -1.40 [-2.87, +0.09] |
| max | 75.58 | 71.85 | 79.31 | best |
| p95 | 69.72 | 69.86 | 69.58 | -5.86 [-9.12, -2.36] |
| top1pct | 75.53 | 72.07 | 78.99 | -0.05 [-0.86, +0.82] |
| mean | 72.02 | 68.45 | 75.59 | -3.55 [-6.72, -0.27] |

## 5. Pose hypotheses

not separable for rule hyb (kept at H = 6).


## Selected (frozen) configuration

```json
{
  "max_nn": 20,
  "rule": "hyb",
  "views": "V01",
  "pooling": "max",
  "mhr": true
}
```

