# Novelty Roadmap — where the headroom is, and which code to touch

Last benchmark sweep: Sept 2026. Targets to beat (standard 4-sample protocol,
O-AUROC / P-AUROC):

| Method | Real3D-AD | Anomaly-ShapeNet |
|---|---|---|
| Reg3D-AD (NeurIPS'23) | 70.4 / 70.5 | 57.2 / – |
| PointCore (ECCV'24) | 82.9 / 73.1 | – |
| ISMP (AAAI'25) | 76.7 / 83.6 | 75.7 / 69.1 |
| MC3D-AD (IJCAI'25) | 78.2 / 76.8 | 84.2 / 74.8 |
| Simple3D (2025) | 80.4 / 92.3 | 86.0 / 92.9 |
| Reg2Inv (NeurIPS'25) | ~83.9 / ~92.0 | ~86.1 / 88.2 |
| Template3D-AD (IJCAI'25) | **84.4** / – | 86.5 / – |
| PASDF (2025) | 80.2 / 74.5 | **90.0** / – |

**The two headline gaps in the field:**
1. Point-level **AUPR** on Real3D-AD is ~0.19–0.25 at best (vs AUROC ~0.92) —
   precise localization is essentially unsolved.
2. A handful of Real3D-AD categories fail for everyone: **gemstone, starfish,
   duck, toffees** (amorphous / high intra-class variation objects).

---

## Direction A — Better features (`src/ad3d/features.py`)

- `compute_fpfh_ms()` is the descriptor. Swap/extend with: surface curvature
  features, normal-angle histograms, learned descriptors (PointMAE features
  can be added and concatenated — the official Real3D-AD repo has ready
  PointMAE/Point-BERT extractors to borrow).
- Simple3D's repo also contains unused SHOT / Spin / CVFH / shape-context
  extractors — a descriptor-combination study is cheap and publishable.
- **Rotation invariance** is a known weakness of FPFH pipelines (Reg2Inv's
  whole pitch). Rotation-invariant descriptors would help Anomaly-ShapeNet.

## Direction B — Smarter aggregation (`lfsa()` in features.py)

- LFSA is a plain mean over `group_size` neighbors. Try attention-weighted
  aggregation, or multi-resolution grouping (concat fine + coarse group stats:
  mean, std, max).
- `num_group`/`group_size` trade-off is essentially unexplored in the papers.

## Direction C — Memory & distance (`src/ad3d/memory.py`)

- Currently: one bank per category, greedy k-center coreset, Euclidean NN.
  Try: per-prototype banks with min-over-banks scoring (a cheap proxy for
  registration), Mahalanobis / local-density-normalized distances,
  PatchCore-style reweighting of the memory.
- **Unified multi-class model**: one shared bank for all 40 categories +
  category-agnostic scoring. Almost nobody reports this; big novelty
  surface + practical value.

## Direction D — Scoring rules (`src/ad3d/scoring.py`)

- Object score is currently `max`. Test top-k mean, percentile stats,
  per-sample z-normalization of the score map (helps when train/test scan
  density differs — exactly the Real3D-AD 360°-vs-single-view gap).
- Score-map smoothing (`smooth_k`) interacts strongly with P-AUPR.

## Direction E — Alignment for Real3D-AD (the biggest single lever)

Train clouds are full 360° scans; test clouds are single-view. Registration-
based methods (Reg3D-AD → ISMP → Reg2Inv → Template3D-AD) own the top of the
O-AUROC table because of this. A cheap experiment: add an ICP pre-alignment
of each test cloud to its nearest training prototype **before** feature
matching (`o3d.pipelines.registration.registration_icp` — no new deps).
Beat Reg2Inv's alignment and you are at SOTA O-AUROC.

**Status: implemented** — `--align icp` (RANSAC+ICP to the mutually aligned
prototypes), `--cuts N` (single-view memory augmentation, the train_cut
trick), `--topk K` (region pooling; helps aliased categories, taxes small
defects), and `--align-poses K` (**multi-hypothesis registration**: symmetric
objects admit several geometrically valid poses; picking the wrong one
inflates normal clouds' scores. Candidates come from perturbation restarts
+ short ICP; selection is geometric (trimmed point-to-surface residual) —
never score-based, which adversarially hides defects). Ablation grid: see
`real3d_ablation.csv` notes in the session — cuts +1.2, icp +2.4, additive.

**Ablation outcome (Real3D-AD, O-AUROC):** baseline 0.637 -> cuts 0.649 ->
icp 0.661 -> icp+cuts 0.673 -> +MHR (`--align-poses 6`) **0.689 = final
Direction-A recipe** (P-AUROC 0.913, P-AUPR 0.366; FPFH only, no training).
Adaptive per-category registration resolution (`--align-voxel 0`, train-side
cut-registration selection) was tried and **REJECTED: 0.671 (-1.8)** — the
train-side proxy does not transfer (prototype-cut registration is too clean
vs real single-view scans; the trimmed residual is biased toward finer voxels
by construction). Per-category oracle resolution would give 0.702 — the
signal exists but is not legitimately capturable (paper discussion point).
Fine voxels do improve point-level precision (pooled P-AUPR 0.227 -> 0.237)
while hurting object-level detection.

## Direction F — Evaluation hygiene (do this regardless)

- Run ≥3 seeds (`--seed`), report mean ± std. Real3D-AD has only ~100 test
  samples/category — margins <1.5% are usually noise.
- Report both point-level conventions (the repo already does).
- Keep a fixed results ledger: `results/*.json` already stores the full config
  with each run — never overwrite, always compare tagged runs.

---

## Suggested first experiment (1 Kaggle session)

1. Run the full baseline on both datasets (`--tag base`).
2. Add ICP pre-alignment (Direction E) behind a config flag → `--tag icp`.
3. Compare per-category CSVs; if gemstone/starfish/duck/toffees improve,
   you have a paper seedling. If not, Direction C per-prototype banks next.

---

## Decision after r6 (2026-10-03) - supersedes "Direction B next"

Facts: Simple3D (handcrafted FPFH-style MSND+LFSA) reaches 80.4 on Real3D-AD;
we reach 68.9 with the same descriptor family, yet beat its Anomaly-ShapeNet
numbers. So the Real3D-AD gap is most likely partial-scan handling /
preprocessing, not representation. Point-MAE is therefore NOT the next step.

Order of work:
1. **Validity (must do):** 3 seeds for base / icp-cuts4 / mhr6. Seed 0 exists only for icp-cuts4 (r3) and mhr6 (r6);
   base must be run for seeds 0,1,2 (notebook cells 3a-3e do this).
   `CONFIGS="base" bash scripts/run_parallel.sh <REAL_ROOT> 0 1 2` and `CONFIGS="icp-cuts4 mhr6" bash scripts/run_parallel.sh <REAL_ROOT> 1 2`
   then `python scripts/aggregate_seeds.py results/`.
2. **Cheap lever:** `--cuts-diverse` (legacy cuts reused one set of directions
   for all prototypes). `CONFIGS="icp-cuts4-div mhr6-div" bash scripts/run_parallel.sh <REAL_ROOT> 0 1 2`.
3. **Gate:** if mhr6-div >= mhr6 + 1.5 pts (beyond 2 std) -> adopt, then explore
   cut realism (match test coverage). If <= noise -> keep mhr6 as final.
4. **Only then Direction B**, as a *fused* FPFH+Point-MAE ablation, time-boxed
   (1 week). Thesis stands without it: the contribution is the diagnosed,
   ablated registration+cuts+MHR chain plus SOTA-class Anomaly-ShapeNet and
   P-AUROC results.

---

## Decision after 3c (2026-10-04)

3-seed result: icp+cuts is the real effect (+4.0 O, +2.9 P-AUROC, +6.1 P-AUPR). MHR adds a small,
consistent gain (+1.1 O, not significant at n=3; +0.4/+0.6 point-level, significant). Per-category MHR
stories from seed 0 do not hold. Frame MHR honestly as a low-cost refinement, not the headline.

Next, in order:
1. **ShapeNet seeds 1,2** (notebook cell 3f, ~1-2.5 h, unmeasured): the "beats published Simple3D" claim is
   currently two identical seed-0 runs. Cheap and necessary.
2. **Diverse-cuts probe: mhr6-div only, seeds 0-2, SPLIT=1** (cell 3d, ~6 h). Adopt only if paired gain over mhr6
   >= +1.5 pts and positive in 3/3 seeds.
3. **Registration robustness** (~48% of test clouds fall back to unfiltered correspondences): log fallback rate and
   fitness per category, then try a more robust initialisation. This is the best remaining lever on object-level AUROC.
4. Direction B (Point-MAE) stays optional/time-boxed; Simple3D is handcrafted-feature, so representation is not the
   proven bottleneck.

## PRIORITY 0 (2026-10-04): feature-scale test before anything else
Simple3D uses FPFH neighbourhoods 40/80/120; we use 100/200/300 and beat the paper on ShapeNet but trail it by ~12 pts
on Real3D-AD, almost entirely on car/seahorse/chicken/airplane. Run `base-nn40`, `base-nn60` (cell 3g, seed 0, ~2.5 h).
If base jumps well above 0.637 -> adopt the better scale for the aligned recipe (`mhr6-nn40`, 3 seeds) and re-check ShapeNet.
If not -> the gap is preprocessing (voxel / points budget / train_cut) - inspect Simple3D's code before registration work.
