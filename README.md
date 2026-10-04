# ad3d — 3D Point-Cloud Anomaly Detection Workbench

A clean, dependency-light, **Kaggle-T4-ready** codebase for benchmarking and
building new methods on the two pure point-cloud anomaly detection datasets:

| Dataset | Categories | Protocol | Data |
|---|---|---|---|
| **Anomaly-ShapeNet** (CVPR'24) | 40 (synthetic) | 4 normal train samples/class | [HuggingFace `Chopper233/Anomaly-ShapeNet`](https://huggingface.co/datasets/Chopper233/Anomaly-ShapeNet) |
| **Real3D-AD** (NeurIPS'23) | 12 (real, high-res) | 4 normal train samples/class; train = 360° scans, test = single-view | [Google Drive pcd zip](https://drive.google.com/file/d/1oM4qjhlIMsQc_wiFIFIVBvuuR8nyk2k0/view?usp=sharing) / [Baidu (code: vrmi)](https://pan.baidu.com/s/1orQY3DjR6Z0wazMNPysShQ) |

The bundled baseline is **Simple3D-lite** — a faithful reimplementation of the
Simple3D recipe (currently the strongest efficient method family on these
benchmarks): multi-scale FPFH → FPS grouping + LFSA aggregation → prototype
memory bank with greedy coreset → nearest-neighbor scoring.

```
Simple3D-lite pipeline          (paper reference numbers)
                       Anomaly-ShapeNet   Real3D-AD
  Simple3D (paper)     86.0 / 92.9 O/P    80.4 / 92.3 O/P
```
Your numbers with this reimplementation will differ somewhat (paper used its
own preprocessed cuts); treat *your own reproduced numbers* as the reference
line to beat — that is good benchmark hygiene anyway.

## Repository layout

```
ad3d/
├── run.py                     # CLI: run the benchmark on a dataset
├── requirements.txt
├── kaggle_ad3d.ipynb          # one-click Kaggle notebook (data + runs)
├── src/ad3d/
│   ├── datasets.py            # loaders for both datasets (official layouts)
│   ├── features.py            # [EXTENSION POINT 1] MSND-FPFH, FPS, LFSA
│   ├── memory.py              # [EXTENSION POINT 2] memory bank + coreset
│   ├── scoring.py             # [EXTENSION POINT 3] point/object scoring
│   ├── metrics.py             # O-AUROC / P-AUROC / AUPR (both conventions)
│   └── method.py              # Simple3DLite: fit/evaluate + run_benchmark
├── tests/
│   └── synthetic_smoke_test.py  # end-to-end test, no data download needed
└── NOVELTY_ROADMAP.md         # where the SOTA headroom is + what to modify
```

## Quickstart (local or Kaggle)

```bash
pip install -r requirements.txt          # torch / open3d / scikit-learn / pandas
python tests/synthetic_smoke_test.py     # verify the pipeline end-to-end (~30 s)

# quick sanity run on 2 categories
python run.py --dataset shapenet --data-root <path-to-pcd-dir> \
    --classes ashtray0,bowl0 --num-group 512 --group-size 64 --tag smoke

# full benchmarks (Simple3D-like settings)
python run.py --dataset shapenet --data-root <path-to-pcd-dir> --tag baseline
python run.py --dataset real3d  --data-root <path-with-12-category-dirs> --tag baseline

# Real3D-AD ablations for the 360°-train vs single-view-test gap:
#   --align icp   RANSAC(FPFH)+ICP-register every test cloud to the prototypes
#   --align-poses K  multi-hypothesis registration: also consider K symmetry-
#                   equivalent poses and keep the best geometric fit (for
#                   symmetric objects like starfish; K=1 disables)
#   --cuts N      augment the memory with N simulated single-view cuts/prototype
#   --topk K      object score = mean of top-K point scores (vs max at K=1)
python run.py --dataset real3d --data-root <path> --align icp --cuts 4 --topk 32 --tag aligned
```

Results are written to `results/<tag>_<dataset>_<timestamp>.csv|.json`
(per-category rows + `__mean__` row).

On **Kaggle**: open `kaggle_ad3d.ipynb`, enable the T4 GPU, Run All.
Expected wall-clock: Anomaly-ShapeNet ≈ 30–60 min, Real3D-AD ≈ 2–4 h
(Real3D-AD clouds are much bigger; FPFH runs on CPU, FPS/kNN on GPU).

## Metrics conventions (important!)

Papers on these benchmarks use different point-level protocols. This repo
reports **both**:

- `p_auroc` / `p_aupr` — computed **per anomalous sample**, then averaged
  (M3DM / MVTec-3D / Real3D-AD benchmark convention — the one behind most
  published tables);
- `p_auroc_pooled` / `p_aupr_pooled` — all points of a category concatenated.

Always state which one you report. Object-level (`o_auroc`, `o_aupr`) is
unambiguous.

## Data layouts expected

```
Anomaly-ShapeNet (<data-root> = the extracted `pcd` directory)
    pcd/<category>/train/*.pcd          4 normal prototypes
    pcd/<category>/test/*positive*.pcd  normal test samples
    pcd/<category>/GT/*.txt             anomalous: "x,y,z,flag"

Real3D-AD (<data-root> = directory containing the 12 category folders)
    <category>/train/*.pcd              4 normal prototypes
    <category>/test/*.pcd               'good' in the name => normal
    <category>/gt/<stem>.txt            anomalous: "x y z flag"
```

## Adding your own method

Subclass and override — see `NOVELTY_ROADMAP.md` for concrete, high-headroom
ideas mapped to exact functions:

```python
from ad3d.method import Config, Simple3DLite

class MyMethod(Simple3DLite):
    name = "MyMethod"
    def sample_features(self, points):
        ...  # your features here
        return centers, center_feats
```

## Notebook convention (Kaggle)
`kaggle_ad3d.ipynb` always has **only the cells of the current step active**; every other code cell is commented out,
so a session is simply **Run All**. The banner cell at the top names the current step and lists the active cells.
Cell 2 clones the repo once and `git pull`s it afterwards, so a pushed update is picked up automatically.
To switch steps: `python scripts/nb_activate.py kaggle_ad3d.ipynb "<STEP label>" "<cell prefix>" ...`
