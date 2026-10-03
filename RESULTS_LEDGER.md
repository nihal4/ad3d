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
| icp-cuts4 | 0 only (r3) | 0.673 | 0.909 | 0.361 | 0.218 |
| mhr6 | 0 only (r6) | 0.689 | 0.913 | 0.366 | 0.227 |

- base-s0 reproduces the earlier reference (0.6370) EXACTLY -> pipeline is deterministic and the
  cuts_diverse refactor did not change default behaviour.
- Seed noise on the 12-cat mean is ~0.6 pts O-AUROC; per-category noise is 1-3 pts
  (shell 0.32-0.38, starfish 0.61-0.67, duck 0.72-0.77). Per-category claims need multi-seed support.
- icp-cuts4 = +3.7 pts and mhr6 = +5.3 pts over base mean (about 6x and 9x the base seed std):
  headline gain very likely real. mhr6 vs icp-cuts4 (+1.6) is only ~2 std -> NEEDS seeds 1,2 (3c).
- Timing (Kaggle T4x2, 4 vCPU): 2 concurrent base jobs ~2h15 each; the same job ALONE ~1h25.
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
