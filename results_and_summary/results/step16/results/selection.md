# Normal-only configuration selection (no test labels used)

Development files: 4; scales: [6, 10]; datasets: ['real3d']. Tie margin 0.5 AUROC points -> simpler option.


## 1. Descriptor base scale (rule loc10, pooling p99)

| option | mean | real3d |
|---|---|---|
| k=6 | 70.25 | 70.25 |
| k=10 | 73.38 | 73.38 |

## 2. Matching rule (k=10, pooling p99)

| option | mean | real3d |
|---|---|---|
| glob_cs | 65.51 | 65.51 |
| glob_full | 64.81 | 64.81 |
| loc10 | 73.38 | 73.38 |
| loc05 | 74.94 | 74.94 |
| loc15 | 72.05 | 72.05 |
| knn64 | 75.29 | 75.29 |
| knn256 | 74.83 | 74.83 |
| soft0p5 | 72.28 | 72.28 |
| soft1 | 72.69 | 72.69 |
| soft2 | 73.44 | 73.44 |
| cal10_k0 | 63.95 | 63.95 |
| cal10 | 72.40 | 72.40 |
| hyb | 72.92 | 72.92 |

## 3. Training views (rule family loc10, pooling p99)

| option | mean | real3d |
|---|---|---|
| V0 | 71.53 | 71.53 |
| V01 | 70.83 | 70.83 |
| V02 | 73.09 | 73.09 |
| M2 | 73.32 | 73.32 |
| V012 | 73.38 | 73.38 |

## 4. Object-score pooling (rule loc05)

| option | mean | real3d |
|---|---|---|
| p99 | 74.94 | 74.94 |
| max | 78.59 | 78.59 |
| p95 | 64.47 | 64.47 |
| top1pct | 77.37 | 77.37 |
| mean | 64.53 | 64.53 |

## 5. Pose hypotheses

not separable for rule loc05 (kept at H = 6).


## Selected (frozen) configuration

```json
{
  "max_nn": 10,
  "rule": "loc05",
  "views": "V02",
  "pooling": "max",
  "mhr": true
}
```

