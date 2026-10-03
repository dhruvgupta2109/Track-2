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
