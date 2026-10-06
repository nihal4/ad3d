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
- 2026-10-04 21:20: GLFM "Cut Training Data" (Drive 1l6jF5nrzgw-6EgjRjGw6071l1CyF_ep2, 1.07 GB zip) = `Real3D-mvtec/<cls>/{train,test}`
  in MVTec-3D format (xyz/*.tiff + rgb/gt png). TRAIN: 8 cuts per category (airplane, shell: 4; duck: 9).
  Its TEST split is smaller (~49 per category: e.g. airplane good 24 / bulge 20 / sink 5) than the official 100 -> NOT used;
  we always test on the official Real3D-AD test set. Loader: datasets.find_train_cut_paths supports train/good/xyz/*.tiff.

## STEP 4 result (2026-10-04 22:40): GLFM/Simple3D cut training data, seed 0 (official test set)
| config | O max | O mean | O p99 | O top200 | P-AUROC | P-AUROC pooled | P-AUPR |
|---|---|---|---|---|---|---|---|
| cut-nn10 (cuts only, no reg.) | 66.9 | 71.6 | 69.4 | 68.2 | 89.0 | 89.7 | 0.340 |
| cut-mhr6-nn10 (cuts registered + MHR) | 74.2 | 72.4 | 77.3 | 76.2 | 91.7 | 92.1 | 0.437 |
| ref: base-nn10 (360 protos) | 74.4 | - | - | - | 90.5 | - | 0.399 |
| ref: mhr6-nn10 (360 protos + sim. cuts) | 78.3 | - | - | - | 92.3 | 93.0 | 0.431 |
- HYPOTHESIS REJECTED: the curated single-view cut data does NOT explain Simple3D's edge; our 360-degree prototypes
  + simulated cuts + registration beat it by 4-7.5 pts (O, max rule). Thesis point: protocol tested, not the cause.
- Even with Simple3D's data AND its mean rule we get 71.6, far from its 80.4 -> remaining suspects: resolution
  (their absolute voxel 0.15, no point cap; we cap 100k at relative voxel 0.01), 4096 groups, 5% coreset, max_nn 40.
- OBJECT-SCORE RULE is a large lever: p99 beats max by +2.5 / +3.1 on both runs; per-category effects are huge
  (cut-mhr6: seahorse 64.2 max -> 83.4 p95; chicken 65.5 max -> 84.0 mean). Must be re-checked on mhr6-nn10 (5a re-run
  saves scores). Rule choice = test-selected; pre-commit to ONE global rule and validate on Anomaly-ShapeNet.
- Next: 5a = mhr6-nn10 (re-run for scores) + mhr6-nn10-pluscut (360 protos + sim. cuts + real cuts).
- results_4.zip saved (results_and_summary/results/cut-*, incl. _scores.json). Cut registration fitness in cut-mhr6-nn10:
  airplane/candybar/diamond/fish/toffees 1.00, duck 0.98, starfish 0.97, shell 0.91, car 0.86, chicken 0.82,
  gemstone 0.75, seahorse 0.73 -> poorly registered training cuts pollute the memory in exactly the categories where
  cut-mhr6 trails mhr6-nn10 (car 58.4 vs 80.7, chicken 65.5 vs 70.4, seahorse 64.2 vs 55.1 is the exception).
  If pluscut (5a) helps, gate real cuts by registration fitness (e.g. keep only fit >= 0.9).
- Notebook convention adopted (2026-10-04): only current-step cells active; scripts/nb_activate.py. Current: STEP 5.

## STEP 5 result (2026-10-05 02:00): real cuts ADDED to our memory help; p99 rule confirmed; config FROZEN
| config (seed 0) | O max | O p99 | O top1pct | O top200 | P-AUROC | P-AUROC pooled | P-AUPR | P-AUPR pooled |
|---|---|---|---|---|---|---|---|---|
| mhr6-nn10 (re-run) | 77.9 | 79.5 | 79.4 | 79.3 | 92.2 | 93.0 | 0.435 | 0.328 |
| mhr6-nn10-pluscut (+ real GLFM cuts) | 80.1 | **81.4** | 81.5 | 81.5 | **93.0** | **93.6** | **0.472** | **0.374** |
- p99 was named as the preferred rule BEFORE these two runs existed; it beats max by +1.6 / +1.3 here (and +2.5 / +3.1 on
  the step-4 runs) -> confirmed on runs not used to choose it (still the same Real3D-AD test set: report as such).
- pluscut vs mhr6-nn10 (O max): seahorse 54.9 -> 75.0 (+20.1), toffees +6.1, gemstone +3.6, airplane +2.3; car 80.7 -> 70.7 (-10.0),
  duck -2.8. Real-cut registration fitness: car 0.92, chicken 0.81, gemstone 0.75, seahorse 0.85 (others >= 0.95).
- NON-DETERMINISM: the mhr6-nn10 re-run gave 77.88 vs 78.31 originally. Categories with fitness 1.00 and no pose switch
  (candybar, car, fish) reproduce exactly; ambiguous ones move 1-3.5 pts (chicken -3.5, toffees -2.7, duck +2.4).
  Open3D RANSAC is not bit-reproducible -> single registered runs carry ~+/-0.4 mean noise; seeds required.
- FROZEN FINAL CONFIG = `mhr6-nn10-pluscut-p99`: 360-degree prototypes + 4 simulated cuts/prototype + GLFM real cuts
  (registered to the prototypes), RANSAC+ICP + MHR(6 poses), MSND max_nn 10 (10/20/30), LFSA 2048x128, coreset 10%,
  object score = 99th percentile of the point-score map. Reference rows: mhr6-nn10-p99, base-nn10-p99.
- Seed-0 CSVs for mhr6-nn10-p99 / mhr6-nn10-pluscut-p99 are DERIVED offline from the step-5 _scores.json
  (results_and_summary/results/*-p99-s0_real3d_20261004-000000.*); point metrics are unchanged by the rule.
- STEP 6 (running): seeds 1,2 for the two registered configs + base-nn10-p99 seeds 0,1,2.
- Protocol note for the thesis: the final config uses the GLFM-released cut views of the SAME four training prototypes
  (no extra objects); report results with and without them (mhr6-nn10-p99 is the no-external-cuts row).

## PRE-REGISTRATION (2026-10-06 19:50, written BEFORE any step-7 result exists): multi-scale fusion
Hypothesis: categories prefer different descriptor scales (base sweep: diamond nn100 +7.3, gemstone nn40 +10, seahorse nn40 +5.8;
with registration diamond/gemstone/seahorse still prefer larger scales by 5-7.5). Fusing scales recovers part of this.
- Runs: final config `pluscut-p99` at max_nn 10 / 40 / 100, seed 0 (step 7, notebook cells 7a-7c).
- PRIMARY rule (fixed now): for each category and scale, object score = p99 of the point-score map, z-normalised with the
  TRAINING reference of that scale (mean/std of the training features' distances to the coreset, saved as train_ref_*);
  fused score = equal-weight mean over the three scales {10, 40, 100}. No test labels, no per-category weights/choices.
- Secondary (reported, not used to decide): ratio normalisation, raw scores, and all 2-scale subsets.
- Decision gate: adopt fusion only if PRIMARY fused O-AUROC >= nn10 run of the same step + 1.5 (beyond the ~0.4 run noise).
  If adopted: implement fusion inside the pipeline (point-level too), then step 6 seeds on the fused config.
- Disclosure for the thesis: the scale set {10,40,100} was informed by the Real3D-AD test-set sweep (selection bias).
- Code: MemoryBank.ref (training distances to coreset), method.evaluate -> train_ref_* metrics, scripts/fuse_scales.py.
- 2026-10-06 20:06: STEP 7 (multi-scale fusion) CANCELLED by decision (cost ~4 GPU-h vs. an uncertain ~+1.5 gain).
  The pre-registration above stays on record; it was never run. Code (MemoryBank.ref, fuse_scales.py, nn40/nn100 configs)
  kept for possible future work. Next: STEP 6 seeds on the frozen final config.

## STEP 8 (planned 2026-10-06 20:12): one-at-a-time boost screens on the 5 weakest categories
- Policy (user decision): beat the benchmark first, seeds after; test ONE change at a time; screen on the 5 weakest
  categories (airplane, car, chicken, duck, shell) to save GPU time; a winner is applied to ALL categories (no per-category
  choice) and validated on all 12 before adoption.
- 8 = Simple3D grouping: final config + --num-group 4096 --coreset 0.05. Reference (pluscut-p99, s0) 5-cat O mean 70.78.
  Gate fixed in advance: 5-cat mean >= 72.8 (+2.0). Next in line if it fails: higher resolution.
- Code: run_parallel.sh SPLIT mode accepts CLASSES=<subset> (merge with --allow-partial).

## STEP 8 result (2026-10-06 21:00): Simple3D grouping (4096 groups, 5% coreset) -> FAIL, dropped
5 weakest categories, seed 0 (O-AUROC ref -> g4096): airplane 74.1->72.8, car 69.7->68.8, chicken 75.1->76.4,
duck 75.3->82.3, shell 59.7->59.5; mean 70.78 -> 71.96 (+1.18 < +2.0 gate). P-AUROC mean 91.58 -> 91.34.
Gain is mostly duck (+7.0), the category that moved +2.4 between two IDENTICAL runs (registration non-determinism);
the other four are within +/-1.3 noise. Not adopted.
- Point counts after preprocessing (pluscut s0 scores): 12k-42k per test cloud (seahorse ~12k), cap of 100k NEVER binds;
  anomalies cover ~330-1500 points (~1-3%). Simple3D's absolute voxel 0.15 ~= 0.0075 in our unit-normalised frame (~2x density).
## STEP 9 (planned 2026-10-06 21:05): higher resolution, ONE change
- voxel 0.01 -> 0.007 (~2x points) with max_nn 10 -> 20 to keep the same PHYSICAL neighbourhood; all else = final config.
  5 weakest categories; gate fixed in advance: 5-cat O mean >= 72.8 (ref 70.78).

## STEP 9 result (2026-10-06 21:45): higher resolution (voxel 0.007, max_nn 20) -> FAIL, dropped
5 weakest categories, seed 0 (O-AUROC ref -> hires): airplane 74.1->60.7, car 69.7->70.2, chicken 75.1->74.3,
duck 75.3->69.8, shell 59.7->50.3; mean 70.78 -> 65.06 (-5.72). P-AUROC 91.58 -> 90.5.
Likely cause: LFSA groups (2048 x 128 points) shrink to ~half the surface area at 2x density -> noisier, more local
aggregation; a fair resolution gain would need re-tuned grouping (coupled multi-parameter search) - not pursued.
## Boost search closed (2026-10-06): both pre-gated boosts negative (grouping +1.18 < +2.0; resolution -5.72).
Decision: keep the frozen final config `mhr6-nn10-pluscut-p99` (81.4 seed 0) and run STEP 6 seeds.
Thesis material: two clean, pre-registered negative results (resolution and Simple3D grouping do not transfer).

## PRE-REGISTRATION (2026-10-06 22:45, BEFORE any real result): NEW METHOD - Prototype Tolerance Field (PTF) fusion
Motivation (evidence): Template3D-AD (IJCAI'25, verified from the paper: single template = first training sample,
RANSAC+ICP, score = coordinate distance x feature differences, NO model of normal variation) is complementary to us per
category: they win shell (+32.4), car (+18.3), fish (+12.5), seahorse (+8.8); we win gemstone (+19.7), candybar (+11.5),
starfish (+6.6), duck (+4.4). Per-category max of the two = 88.1 (oracle, not a result).
Method (src/ad3d/geometry.py, --geo fuse):
 1. aligned prototypes (registration stage) -> point-to-plane ICP refinement into one frame (voxel 0.005);
 2. leave-one-out residual of every prototype point to the union of the OTHER prototypes = normal instance variation;
 3. tolerance field sigma(x) = LOO residuals averaged over 16 nearest surface points; floor eps = median sigma;
 4. test point (after MHR registration + small point-to-plane ICP): residual r to the merged surface
    (point-to-plane + tangential excess), u = r / (sigma(nearest) + eps), averaged over 12 nearest test points;
 5. fused point score = feature score x (1 + u)   [scale-free product: no tuned weight]; object score = p99.
Same run reports: features only, PTF only, residual with ONE global threshold (Template3D-AD-like), and that fused
-> isolates the contribution of the tolerance field itself.
- PRIMARY: fused (PTF) O-AUROC, 12-category mean, seed 0 (step 10).
- Gate 1 (adopt): fused >= features-only of the SAME run + 1.5.   Gate 2 (SOTA claim candidate): fused > 84.4,
  then confirmed with seeds 1,2 (step 6 updated to the fused config) before any claim.
- Nothing is tuned on this run: voxel 0.005, k 16/12, product fusion, p99 are fixed now.
- Synthetic sanity check (toy ellipsoid with a naturally variable region, small bumps): features 58.2, uniform residual
  69.5, PTF 73.8, fused 73.4 (O-AUROC); point-level PTF 78.7 < uniform 87.5 (tolerance also damps true defects inside
  variable regions) - a known trade-off to watch on real data.

## STEP 10 result (2026-10-06 23:50): Prototype Tolerance Field fusion -> FAILS both gates, NOT adopted
12 categories, seed 0, one run (all channels from the same run):
| O-AUROC | features | geo uniform | geo PTF | fused uniform | FUSED PTF | Template3D-AD |
|---|---|---|---|---|---|---|
| mean | 82.0 | 74.4 | 73.8 | 79.1 | 78.6 | 84.4 |
P-AUROC: 93.1 / 83.4 / 81.2 / 91.5 / 91.1.  P-AUPR: 47.4 / 49.7 / 45.3 / 59.0 / 56.6.
- Gate 1 (fused >= features + 1.5): FAIL (78.57 vs 81.98). Gate 2 (> 84.4): FAIL.
- Tolerance field vs one global threshold: no gain (geo 73.8 vs 74.4; P-AUROC 81.2 vs 83.4) -> NEGATIVE for the PTF idea.
- Geometric channel is category-specific: seahorse 100.0 (all geo variants), fish 93-96.5, diamond fused 99.7; but airplane
  51-54, gemstone 57, chicken 61, candybar 76. Our geometry does NOT reproduce Template3D-AD on shell (53 vs 92.1) or car (71 vs 88).
- Localisation: fused-uniform P-AUPR 59.0 vs features 47.4 (+11.6) with P-AUROC -1.6 (observation, not the pre-registered goal).
- Ceiling: per-category oracle over {features, geo, geou, fused, fuseu} = 84.4 (test-label selection, NOT a result) ->
  with the current signals no fair combination can exceed Template3D-AD on Real3D-AD.
- Features-only column (82.0) = same config as mhr6-nn10-pluscut-p99 (81.4 earlier): difference within registration noise.
- Decision: stop the SOTA search on Real3D-AD; consolidate (STEP 6 seeds on the feature config, then Anomaly-ShapeNet).
  No further fusion rules will be tried on this test set (would be post-hoc selection).

## Literature check (2026-10-07) - CORRECTION
- Reg2Inv (NeurIPS 2025) Real3D-AD O-AUROC = 78.0 (from the paper's table; per category: candy 100, diamond 100, fish 67.2,
  airplane 81.8, car 75.8, chicken 94.4, duck 75.0, gemstone 73.5, seahorse 53.2, shell 69.2, starfish 84.1, toffees 62.6),
  P-AUROC 87.8. The "~83.9" in our old tables is its ANOMALY-SHAPENET number. Reg2Inv is BELOW us on Real3D-AD.
- Synthesis4AD (arXiv 2604.04658, Simple3D authors, trained): Real3D-AD O 80.9. Scientific Reports 2026 (Hoang et al.): 78.1.
- arXiv 2609.35059 claims 89.6 on Real3D-AD but in a cross-dataset setting (trained on Anomaly-ShapeNet) - not comparable,
  not peer-reviewed.
- => Template3D-AD (84.4) remains the only verified method above us. Its key difference: it compares each test centre with
  the descriptor at the CORRESPONDING location of the registered template (location-conditioned), not with a global bank.

## PRE-REGISTRATION (2026-10-07 00:20, BEFORE any real result): NEW METHOD - location-aware memory (step 11)
Mechanism (src/ad3d/localmem.py, --local-mem 0.10): every training descriptor (aligned prototypes, simulated cuts,
registered real cuts) keeps the 3D position of its group centre in the registered frame; no coreset. A test descriptor is
compared only with training descriptors whose centres lie within rho of its own centre (fallback: global NN if none).
Rationale: shell 58.9 vs Template3D-AD 92.1 - ridge-like defects match normal ridges elsewhere under global search.
- PRIMARY: rho = 0.10 (unit-normalised frame), object score p99, 12-category O-AUROC mean, seed 0. Fixed now.
- Same run reports: global memory (current method), rho 0.05 and 0.15 (sensitivity, NOT used to choose).
- Gate 1 (adopt): loc(0.10) >= global (same run) + 1.5.  Gate 2 (user target): loc(0.10) >= 85.4 (Template3D-AD + 1),
  then seeds 1,2 before any claim.
- Synthetic check (toy ellipsoid with ridges on one half; defects = ridge patch on the smooth half, or bump):
  global O 0.605 -> loc 0.840; misplaced-ridge defects 0.648 -> 1.000; bumps 0.562 -> 0.680; P-AUPR 0.14 -> 0.48.
