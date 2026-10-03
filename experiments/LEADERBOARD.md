# Public leaderboard log (Track 2)

| date | file | model / experiment | std CV | hnW (hard-neg view) | public PR-AUC | rank |
|---|---|---|---|---|---|---|
| 2026-10-03 | submissions/sub_v3_lr.csv | v3 LR (C=0.1, eng feats, hour cat, 1h->non-burst flag) | 0.2303 | 0.1796 | **0.17206** | #3 (#1 0.19010, #2 0.17310) |
| pending | submissions/sub_v4_lr.csv | v4 = v3 + new_device x log(age) + new_device x night | 0.2411 | 0.1970 | | |
| pending | submissions/sub_v4_lambda05_probe.csv | v4 with transductive correction logit(p) - 0.5 log w(x) (probe only, not packaged) | 0.2298 | (self-ref) | | |

Which local view tracks public? So far one point: covW predicted 0.357, hnW predicted 0.180, std 0.230; public 0.172 -> hnW.
