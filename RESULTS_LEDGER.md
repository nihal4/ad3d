# Results ledger (append-only). Seed 0 unless stated. Updated 2026-10-03.

## Real3D-AD (12 categories, mean of per-category metrics)
| tag | config | O-AUROC | P-AUROC | P-AUPR | pooled P-AUPR | file |
|---|---|---|---|---|---|---|
| base | none | 0.637 | 0.876 | 0.294 | - | (not in repo; per-cat in real3d_ablation.csv) |
| r1 | cuts4 | 0.649 | 0.897 | 0.319 | - | (not in repo) |
| r2 | icp | 0.661 | 0.894 | 0.333 | 0.196 | r2-icp_*.csv |
| r3 | icp+cuts4 | 0.673 | 0.909 | 0.361 | 0.218 | r3-*.csv |
| r4 | r3+topk32 | 0.677 | 0.908 | 0.358 | 0.216 | r4-*.csv |
| r6 | r3+MHR6 (**current best**) | **0.689** | 0.913 | 0.366 | 0.227 | r6-*.csv |
| r7 | MHR6, voxel 0.03, 3 cats only | (airplane +10.0, starfish -5.6 vs r6) | | | | r7-*.csv |
| r8 | r6+auto-voxel | 0.671 | 0.911 | 0.369 | - | (not in repo) NEGATIVE |

## Phase 3 - seeds (Real3D-AD, 12 cats, 1206 test samples each)
| config | seeds | O-AUROC | P-AUROC | P-AUPR | pooled P-AUPR |
|---|---|---|---|---|---|
| base | 0,1,2 | 0.6362 +/- 0.0057 (0.637/0.641/0.630) | 0.8779 +/- 0.0022 | 0.2949 +/- 0.0027 | 0.1911 +/- 0.0029 |
| icp-cuts4 | 0,1,2 | 0.6765 +/- 0.0078 (0.673/0.671/0.685) | 0.9066 +/- 0.0024 | 0.3563 +/- 0.0050 | 0.2174 +/- 0.0022 |
| mhr6 | 0,1,2 | 0.6873 +/- 0.0114 (0.689/0.675/0.698) | 0.9107 +/- 0.0028 | 0.3620 +/- 0.0049 | 0.2223 +/- 0.0071 |

- base-s0 reproduces the earlier reference (0.6370) EXACTLY -> pipeline is deterministic and the
  cuts_diverse refactor did not change default behaviour.
- Seed noise on the 12-cat mean is ~0.6 pts O-AUROC; per-category noise is 1-3 pts
  (shell 0.32-0.38, starfish 0.61-0.67, duck 0.72-0.77). Per-category claims need multi-seed support.
- icp-cuts4 = +3.7 pts and mhr6 = +5.3 pts over base mean (about 6x and 9x the base seed std):
  headline gain very likely real. mhr6 vs icp-cuts4 (+1.6) is only ~2 std -> NEEDS seeds 1,2 (3c).
- icp-cuts4 vs base, paired per seed: +4.03 pts (per-seed +3.6/+3.0/+5.5, sd 1.3). P-AUROC +2.9, P-AUPR +6.1 (+21% rel).
  Solid. mhr6 seed 0 (0.6887) is +1.2 over the icp-cuts4 3-seed mean (~1.6 sd): UNRESOLVED until mhr6 seeds 1,2.
- CAUTION per-category: icp-cuts4 is very unstable on starfish (0.576/0.638/0.726), shell (0.567/0.443/0.563), toffees (0.613/0.706/0.682).
  The earlier "MHR recovers starfish +11" claim rests on a LOW seed-0 draw (3-seed mean 0.646; MHR gain only +3.9, ~0.5 sd). Do NOT headline it.
  MHR evidence that survives: duck (+6.3 vs icp-cuts4 mean, ~2.7 sd), candybar (+4.5, ~2.5 sd). Wait for mhr6 seeds.
- icp-cuts4 job time: ~3h55 each with 2 concurrent (21:41 -> 01:36 local). Timing (Kaggle T4x2, 4 vCPU): 2 concurrent base jobs ~2h15 each; the same job ALONE ~1h25.
  So 2-GPU parallelism only gives ~1.25x over sequential (CPU-bound). Do not expect 2x.

## Anomaly-ShapeNet (40 cats): O 0.881/0.882, P-AUROC 0.935, P-AUPR 0.673 (2 runs). CSVs not in repo.

## Caveats on record
- Single seed everywhere. Gains <~1.5 pts (cuts +1.2, topk +0.4) are NOT established.
- P-AUPR 0.366 is per-category mean; pooled is 0.227. State the convention in the paper.
- Simple3D is a HANDCRAFTED-descriptor method (arXiv 2507.07435), not deep-feature.
  So the 68.9 vs 80.4 gap is not shown to be an "FPFH ceiling".
- Legacy `--cuts N` used identical cut directions for every prototype (bug-like).
  All r1-r8 results include that behaviour. `--cuts-diverse` fixes it (new, unrun).

## Pending (in order)
1. base seeds 0,1,2; icp-cuts4 + mhr6 seeds 1,2 (seed 0 = r3/r6 CSVs; clean base seed-0 CSV not in repo)
2. seeds 0,1,2 for: icp-cuts4-div, mhr6-div
3. decision gate -> see NOVELTY_ROADMAP.md "Decision after r9"

## Tooling
- scripts/run_parallel.sh : 2-GPU parallel queue (jobs round-robin, or SPLIT=1 by category)
- scripts/merge_categories.py : merges split runs (verified: re-merging r6 halves reproduces O 0.6887)
- scripts/aggregate_seeds.py : mean +/- std and paired deltas over seeds

## Simple3D reference facts (from arXiv 2507.07435 text as retrieved 2026-10-04 - VERIFY in the PDF before citing)
- FPFH multi-scale neighbourhoods 40/80/120, group size 128 (ours: 100/200/300, 128).
- Real3D-AD per-category O/P-AUROC: car 98.1/99.2, seahorse 93.0/94.2, chicken 82.6/86.1, airplane 76.5/88.1, shell 51.4/71.6.
- Ours (mhr6, 3-seed means): car 54, seahorse 64, chicken 64, airplane 61, shell 53 (O-AUROC).
- => the 12-pt gap is concentrated in car (-44), seahorse (-29), chicken (-18), airplane (-15). Shell is hard for everyone.
- Hypothesis H1: our descriptor scale is too coarse for Real3D-AD (tests: --max-nn 40 / 60). Notebook cell 3g.

## Literature review (2026-10-04) - paper/related_work.tex + paper/references.bib
- 14 peer-reviewed papers, venue/year/pages checked on publisher or proceedings pages.
  EXCLUDED (no peer-reviewed venue found, do not cite as peer-reviewed): PointCore (arXiv 2403.01804),
  3DKeyAD (arXiv 2507.13110), CPMF (arXiv 2303.13194). Check dblp/ECVA before adding any of them.
- CORRECTIONS to earlier comparison tables: PO3AD is CVPR 2025 (not ECCV 2024).
  "PatchCore-FPFH 68.2" is PatchCore (FPFH+Raw) in Real3D-AD Table 4; plain PatchCore (FPFH) is 59.3.
  Reg3D-AD point-level AUROC in Real3D-AD paper is 0.700 (roadmap table says 70.5) - recheck.
- Verified numbers: Reg3D-AD 70.4 O; R3D-AD 73.4/74.9 O; Template3D-AD 84.4/86.5 O; PASDF 80.2/90.0 O.
  Simple3D 80.4 and Reg2Inv ~83.9 NOT re-verified against PDFs (only abstracts read).
- Open TODOs in related_work.tex: confirm "single-run reporting" claim (G3) in each paper; fill 3g
  feature-scale result; 3-seed ShapeNet before claiming it beats Simple3D.

## 3g result (2026-10-04): FEATURE SCALE MATTERS (Real3D-AD, base, seed 0, single run each)
| max-nn | O-AUROC | P-AUROC | P-AUPR (per-cat) | P-AUPR pooled |
|---|---|---|---|---|
| 100 (base-s0) | 0.6370 | 0.8757 | 0.294 | - |
| 60  | 0.6581 | 0.8900 | 0.330 | 0.231 |
| 40  | 0.6988 | 0.8975 | 0.363 | 0.268 |
- Monotone in scale. nn40 alone (+6.2 O, no registration) already beats the 3-seed mhr6 (0.687); base seed sd is 0.6, so real.
- Biggest nn40 gains vs nn100: fish +17.5, candybar +15.6, airplane +14.4, gemstone +12, toffees +10, starfish +12.
- NOT fixed by scale: car 50 (Simple3D 98.1), seahorse 58 (93.0), shell 32 (51.4), chicken 66 (82.6). Car/seahorse are
  likely preprocessing/normalisation/partial-view effects, not descriptor scale. Shell got worse (38 -> 32).
- Open: optimum may be below 40 (3i: nn30, nn20); does registration still add at nn40? (3j). Single seed: confirm with seeds 1,2 at the chosen scale.
- Impact on thesis claim: the registration chain's gain was measured at a mis-scaled descriptor; must be re-measured at the new scale.

## 3i result (2026-10-04): scale sweep continues to help (Real3D-AD, base, seed 0, single runs)
| max-nn (x[1,2,3]) | O-AUROC | P-AUROC | P-AUPR per-cat | P-AUPR pooled |
|---|---|---|---|---|
| 100 | 0.637 | 0.876 | 0.294 | 0.192 |
| 60  | 0.658 | 0.890 | 0.330 | 0.231 |
| 40  | 0.699 | 0.898 | 0.363 | 0.268 |
| 30  | 0.713 | 0.902 | 0.378 | 0.283 |
| 20  | 0.725 | 0.903 | 0.390 | 0.292 |
- Radius is 1e6, so max_nn = neighbourhood size; Simple3D's 40/80/120 is already beaten by 20/40/60 here.
- Not saturated in O-AUROC/AUPR; P-AUROC flattening. The 30->20 gain (+1.2 O) is ~2 base-seed sd: suggestive, not firm.
- Category optima differ: airplane, candybar, car, fish, toffees, shell keep improving; diamond (99.0->95.4),
  gemstone, starfish (73.6@60 -> 59.8@20), seahorse peak at larger scale. Mean hides this -> per-category or multi-scale-set
  selection is a possible contribution (but per-category oracle selection is not legitimate; needs a train-side criterion).
- Car improved only at small scale (50 -> 66), still far from Simple3D 98.1. Seahorse 53, chicken 65 unchanged.
- Next (3j): base-nn10 (is there a floor?) and mhr6-nn20 (does registration still add?). Then seeds 1,2 on the winners.

## 3j result (2026-10-04): Real3D-AD, seed 0, single runs
| config | O-AUROC | P-AUROC | P-AUPR per-cat | P-AUPR pooled |
|---|---|---|---|---|
| base-nn20 | 0.725 | 0.903 | 0.390 | 0.292 |
| base-nn10 | 0.744 | 0.905 | 0.399 | 0.304 |
| mhr6-nn20 | 0.751 | 0.923 | 0.430 | 0.323 |
| (mhr6-nn100, 3-seed) | 0.687 | 0.911 | 0.362 | 0.222 |
- No floor in scale yet (nn10 best base). base-nn10 (74.4, no registration) ~ mhr6-nn20 (75.1): at small scale registration adds only +2.6 O
  (vs ~+5 at nn100), but +2.0 P-AUROC and +4 P-AUPR: registration's main value is point-level.
- Registration is category-dependent at nn20: HELPS shell +21.7, chicken +6, gemstone +6, duck +5, starfish +13; HURTS airplane -20.5
  (82.0 -> 61.5; also 61.9 at nn100: slender-object registration failure), toffees -9, fish -2. Single seed: only large effects trustworthy.
- mhr6-nn20 P-AUROC 0.923 equals Simple3D's reported 92.3 (convention to be checked).
- 591 'too few correspondences/fall back' log lines in mhr6-nn20 -> registration robustness remains a lever (airplane).
- SELECTION BIAS: scale and config were chosen on Real3D-AD test results (6 scales, several configs). Use ShapeNet as an
  independent check of the scale, and report the selection procedure in the thesis.
- Next (3k): mhr6-nn10, base-nn6; then seeds 1,2 for the final base and mhr6; ShapeNet 3 seeds at the final scale.

## 3k result (2026-10-04): FINAL CONFIG candidate = mhr6-nn10 (Real3D-AD, seed 0, single run)
| config | O-AUROC | P-AUROC | P-AUPR per-cat | P-AUPR pooled |
|---|---|---|---|---|
| base-nn6  | 0.729 | 0.903 | 0.405 | 0.296 |
| base-nn10 | 0.744 | 0.905 | 0.399 | 0.304 |
| mhr6-nn10 | **0.783** | **0.923** | **0.431** | **0.327** |
- Scale floor at nn10 (nn6 worse; normals use 10 neighbours). Registration adds +3.9 O at nn10 (base-nn10 -> mhr6-nn10), +1.8 P-AUROC, +3.2 P-AUPR.
- mhr6-nn10 per category O-AUROC: airplane 75.1, candybar 97.7, car 80.7, chicken 70.4, diamond 93.6, duck 80.2, fish 86.9,
  gemstone 73.1, seahorse 55.1, shell 67.0, starfish 79.4, toffees 80.5. Airplane no longer collapses (82->61 seen at nn20).
- vs Simple3D published: shell 67.0 vs 51.4 (ours better); airplane 75.1 vs 76.5 (par); chicken 70.4 vs 82.6; car 80.7 vs 98.1;
  seahorse 55.1 vs 93.0 (largest remaining gaps: seahorse, car, chicken).
- Registration diagnostics (mhr6-nn10): mean fitness 0.967 (lowest: gemstone 0.89, shell 0.90, chicken 0.93); pose-switch 52% diamond, 27% starfish, 13% candybar/shell.
- SELECTION BIAS: scale/config chosen on Real3D-AD test; seeds 1,2 reduce seed noise but not that bias. ShapeNet (3f) is the independent check.
- Pending: seeds 1,2 for mhr6-nn10 and base-nn10 (3l); ShapeNet base vs base-nn10 x3 seeds (3f); then set Config.max_nn default.

## Plan change (2026-10-04, user decision): improve first, seeds once on the frozen config
- scripts/screen.sh: one-factor screens on car, seahorse, chicken (base-nn10 reference already on disk). Presets: g64 g256
  ng1024 ng4096 sm6 sm24 cs20 vx005 vx02 top32. Overfitting guard: winners are validated on all 12 categories before adoption.
- Selection-bias note: every test-tuned choice (scale, screens) is reported as such; ShapeNet (3f) is the held-out check.

## Analysis + decision (2026-10-04 evening): why we trail Simple3D on Real3D-AD, and what to run next
### Evidence from Simple3D's OFFICIAL code (github.com/hustCYQ/MiniShift-Simple3D, cloned 2026-10-04)
- data/real3d.py: TRAIN = `real3d_train_cut/<cls>/train_cut/*.asc` (pre-cut single-view clouds), NOT the 4 official 360-degree
  prototypes. Test = their own .txt export with the label column. Voxel: ABSOLUTE 0.15 in raw units, no normalisation, no point cap.
- The cut data is GLFM's "Cut Training Data" release (github.com/hustCYQ/GLFM-Multi-class-3DAD README, Google Drive
  id 1l6jF5nrzgw-6EgjRjGw6071l1CyF_ep2). The official Real3D-AD repo (M-3LAB/Real3D-AD) has NO train_cut data.
- feature_extractors/features.py: object score on Real3D-AD = MEAN of the point-score map (`s = torch.mean(s_map)`);
  ShapeNet = mean of top-80. Coreset 5% (sparse random projection). Point score smoothing: FPS 1024 centres, k=12, mean.
- README command for Real3D-AD: --num_group 4096 --group_size 128 --max_nn 40 --use_MSND --use_LFSA.
  MSND code concatenates [s1,s2] then cat([.., s2, s3]) -> scale 2 is duplicated (40/80/80/120).
- Pixel AUROC = roc_auc_score over ALL points of ALL test samples of a category (= our POOLED convention).
  => Simple3D's 92.3 P-AUROC compares with our pooled P-AUROC: mhr6-nn10 = 93.0 (pooled), i.e. already above.
- CONSEQUENCE: Simple3D's 80.4 O-AUROC uses (a) curated single-view training cuts and (b) a mean object score. Our
  numbers use the official 360-degree prototypes. Comparisons must state the training-data protocol.
### Evidence from our own runs (seed 0)
- Per-category best scale differs (diamond 100, starfish 60, chicken/gemstone/seahorse 40, airplane 20, car/candybar/fish/toffees 10,
  duck/shell 6). Oracle per-category scale = 78.0 (test-selected, NOT legitimate) vs single nn10 = 74.4 -> multi-scale headroom ~3.6.
- Registration gain at nn10 per category: shell +17.0, starfish +7.5, gemstone +7.3, duck +6.8, chicken +6.1, car +3.5,
  seahorse +2.8, diamond +1.9, candybar +0.5, airplane -1.3, fish -1.5, toffees -3.9.
- Seahorse is insensitive to scale (51-58) and registers well (fitness 0.97): its gap is not scale/alignment -> likely training-view mismatch.
### Decision (priority order)
1. STEP 4: train on the published cut data (cut-nn10, cut-mhr6-nn10). Directly tests the hypothesis that the training-view
   protocol explains most of the 2-point gap to Simple3D and the seahorse/car gaps. Code: --train-cut-root (+ cuts registered like test clouds).
2. Object-score rule (max vs mean vs top-k): now computed OFFLINE from *_scores.json (scripts/score_rules.py) - no extra GPU time.
3. Multi-scale descriptor set (e.g. 10/30/90) - principled replacement for per-category scale.
4. The 3m hyper-parameter screen is deprioritised (second-order).
Then: freeze config -> seeds 1,2 -> ShapeNet 3 seeds.
