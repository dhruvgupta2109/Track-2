# Public leaderboard log (Track 2)

| date | file | model / experiment | std CV | hnW (hard-neg view) | public PR-AUC | rank |
|---|---|---|---|---|---|---|
| 2026-10-03 | submissions/sub_v3_lr.csv | v3 LR (C=0.1, eng feats, hour cat, 1h->non-burst flag) | 0.2303 | 0.1796 | **0.17206** | #3 (#1 0.19010, #2 0.17310) |
| 2026-10-03 | submissions/sub_v4_lr.csv | v4 = v3 + new_device x log(age) + new_device x night | 0.2411 | 0.1970 | **0.17966** | |
| 2026-10-03 | submissions/sub_v4_lambda05_probe.csv | v4 with transductive correction logit(p) - 0.5 log w(x) (probe only, not packaged) | 0.2298 | (self-ref) | **0.16822** (retired) | |
| ? | unidentified earlier candidate (probably sub_v1_lr_eng / v2, identical preds) | v1/v2 LR | 0.2286 | 0.1749 | 0.17617 | |

Which local view tracks public? So far one point: covW predicted 0.357, hnW predicted 0.180, std 0.230; public 0.172 -> hnW.

Deltas vs v3: v4 std +0.0108, hnW +0.0174, public +0.0076 (direction agrees; both local views overstate). lambda probe std -0.0113 vs v4, public -0.0114 -> std CV got the direction right, hnW (self-referential) did not.
If 0.17617 is v1/v2: v1->v3 std +0.0017, hnW +0.0047, public -0.0041 (direction disagrees; both within public noise ~0.03).

## Round 5 (v5 search, all vs v4) - no candidate replaces v4
- targeted single interactions (17 tested): best burst_night +0.002 std (CI spans 0); burst_young/cash_burst/big_young significantly worse.
- C sweep 0.03..30: flat 0.05-0.3, C=0.1 kept. Hard-example reweighting x1.25/1.5/2: neutral/worse. Residual LGB on v4 margin: worse.
- v5b burst_night: fresh seeds d_std +0.0022/+0.0012, d_hnW +0.0016/-0.0007; test-like holdout 0.421->0.307, hnW_miss 0.1960->0.1935 -> rejected for packaging.
- optional probe CSV: submissions/sub_v5b_burstnight_probe.csv (tests: are test daytime bursts legit?)

## Round 6-7 (new search space) - candidate v5
| candidate | std (seed 2026) | fresh seeds d_std (99/314/7) | fresh d_hnW | hnW | hnW_miss | holdout / holdout_hnW | file |
|---|---|---|---|---|---|---|---|
| v4 | 0.2411 | - | - | 0.1970 | 0.1960 | 0.421 / 0.278 | sub_v4_lr.csv (public 0.17966) |
| v5 = v4 + age<120 + ctx amount + kNN d_ratio | 0.2490 | +0.0025/+0.0028/+0.0047 | +0.0029/+0.0032/+0.0055 | 0.2070 | 0.2022 | 0.430 / 0.296 | sub_v5_lr_knn.csv |
| v5a = v4 + age<120 + ctx amount | 0.2440 | +0.0027/+0.0019/+0.0014 | +0.0050/+0.0045/+0.0034 | 0.2026 | 0.2011 | 0.427 / 0.290 | sub_v5a_lr.csv |
Rejected this round: isolation forest, legit-density, WoE bins, MoE (nd/young/foreign), ET/RF/HGB, hard-negative specialists, band reranker,
subtype detectors, all stacking except kNN, external augmentation, amount steps, t1>=3.

## Exact public mapping (md5 of CSV, first 10 hex)
| file | md5 | public |
|---|---|---|
| submissions/sub_v1_lr_eng.csv (== sub_v2_lr_hardened.csv, identical bytes) | 9186008608 | 0.17617 (by elimination; v0 never requested - user to confirm) |
| submissions/sub_v3_lr.csv | 43744b16d0 | 0.17206 |
| submissions/sub_v4_lr.csv | aecf532c43 | 0.17966 |
| submissions/sub_v4_lambda05_probe.csv | 72badb25e7 | 0.16822 |
| submissions/sub_v5_lr_knn.csv | 42b34e09bb | 0.18626 (assumed - only new file requested) |
| submissions/sub_v5b_burstnight_probe.csv | 869c1d1923 | unknown / not reported |
Public/std-CV delta ratio (vs v3): v4 0.70, v5 0.76; hnW overstates ~2x; v1 vs v3 public +0.0041 vs expected -0.003 (~1.2 paired sd, noise).
Paired public noise for rank-corr ~0.98 models: sd ~0.002 (12k rows) - 0.006 (6k rows).
