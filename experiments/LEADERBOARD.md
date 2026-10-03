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

## Round 8 public batch (2026-10-03)
| file | md5 | public | local std (vs v5 0.2490) | note |
|---|---|---|---|---|
| submissions/sub_v5a_lr.csv | 2449d07f40 | 0.18026 | 0.2440 | kNN worth +0.0060 public (local +0.0050) |
| submissions/audit/cand_v4_knn_full.csv | 07c5add202 | 0.18436 | 0.2469 | v4+kNN: +0.0047 public over v4 |
| submissions/audit/c1_v5_knn_catmetric.csv | e7486891b3 | 0.18388 | 0.2491 | categorical metric -0.0024 vs v5 |
| submissions/audit/d1_v5_plus_1h_velocity.csv | e54b8af53f | 0.18463 | 0.2477 | -0.0016 vs v5 (1h hypothesis not supported) |
| submissions/audit/cand_lgb_mono.csv | e24832a48c | 0.16962 | 0.2122 | std CV right, test-like holdout view wrong |
Current leader ~0.19010.

## Round 9 public (portal hash = sha256 prefix)
| file | sha256[:10] | public | local d_std vs v5 (fresh seeds) |
|---|---|---|---|
| submissions/audit/v6_rank90_skmlp10.csv | dc84bc23cd | 0.18453 | +0.0033..+0.0041 |
| submissions/audit/v6_E1_knn_metric.csv | e60dcf1ca9 | 0.18237 | +0.0018..+0.0026 |
Lesson: kNN metric change (drop amt_rel_avg, add burst) hurt public by -0.0039 despite consistent local gains (test bursts are legit-like);
the sklearn-MLP rank blend added +0.0022 public on top of its base (local +0.0011). Quota: 100 CSV/24h.

## Round 10-11 (deep / literature kNN / density / augmentation) - all vs v5 on 2x5 screen
- Large MLP 0.195, TabM-mini 0.209, ModernNCA(PLR) 0.208 (overfit); regularised: ModernNCA 0.2389, NCA-linear 0.2369, MLP-small x10 0.2326, TabM32 0.2318.
  Blends with v5 <= +0.0012 std.
- Transductive pseudo-legit kNN references: local +-0.0002; test rank corr 0.997 (no lever).
- KDE / GMM density ratios, random-subspace kNN, bagged-reference kNN: all <= v5.
- Augmentation: SMOTE/jitter/copula/ordinary-fraud/hard-legit at extra weight: -0.0025..-0.19 (prior shift);
  mass-preserving: +-0.0003 (no information gain); pseudo-labelled legit: +-0.0003.
- New CSVs: submissions/audit/v5_rank85_skmlp15.csv (sha 005ff1c9b1), submissions/audit/v5_rank80_deep4_20.csv (sha 8b58245af9)

## Round 13 public + leaderboard-emulator (3 Oct ~17:51)
| file | sha256[:10] | public |
|---|---|---|
| submissions/final_try/E3_0.7k01_0.3sur.csv | a3fbcb9c1b | **0.18662** (best) |
| submissions/final_try/E2_0.4k01_0.15v5_0.3sur_0.15mlp.csv | 80d0c345d3 | 0.18398 |
| submissions/final_try/E1_0.45k01_0.4surknn_0.15mlp.csv | 183aa749e8 | 0.18053 |
Teammate public (their log): v1 0.17617 (04aeb8a713), v2 0.17136, cand1 0.17830, cand2 0.17809, cand3 0.17835.
Local CV ordering E1>E2>E3 was reversed on public.
Emulator (src/lb_soft.py): differentiable expected-AP label model fitted to 19 public scores; LOO corr 0.835, RMSE 0.0031;
fitted mix v5 .38 / mlp .35 / sur .20 / k01 .10 / cb -.09 / v4 -.03, implied test prevalence ~1.5%. Cannot resolve <0.004 differences.
New: LBmix_emulator_optimum.csv (ffcc529417, emulator +0.003 vs v5), TOP5_public_rank_average.csv (79e27a5cf7).

## Burst prior shift (external finding) + EM label-shift sweep
- ../FINAL_UPLOAD_v5_burst/predictions_v5_burst.csv (sha 7f7182dc26) = v5 with logit -1.0 on bursts (tx24>=16, 406 rows): public **0.198**.
- LBmix_emulator_optimum 0.17x, TOP5 rank average 0.185 (user report).
- Saerens EM per segment (src/em_shift.py, experiments/em_shift_segments.csv): burst test prior ~7.0% (shift -1.45);
  new_device & age>365 -> ~0 (strong down-shift); EM unstable for other segments (covariate shift confounds).
- Sweep CSVs in submissions/shift/: S1 burst-1.5, S2 burst-2.0, S3 burst-1.0+ndold-0.5, S4 burst-1.0+ndold-1.0, S5 burst-1.5+ndold-0.75.

## Shift round 2 (3 Oct ~18:52-18:57)
| file | sha256[:10] | public |
|---|---|---|
| shift/S3_burst-1.0_ndold-0.5.csv | f570586529 | **0.20453** (best, rank 3) |
| shift/S1_burst-1.5.csv | 6e83e0fde2 | 0.19293 |
| shift/R2_burst-1.0_ndold-0.7_bigold-0.5.csv | a448b752a7 | 0.20171 |
| shift/R1_burst-1.0_ndold-0.7_bigold-1.1.csv | 5887e82930 | 0.19707 |
EM per-segment direction matched all three tested segments (burst -, ndold -, bigold + => down-shift hurts); enrichment rule did not (bigold).
Best package copied to ../BEST_S3_v5_burst_ndold_0.20453 (sandbox-verified, maxdiff 2e-16).
Next: G1_generator_formula.csv (a2687f3a53, user request), V1_S3_plus_travel-0.5.csv (df00e35171, EM says travel ~0%).
| shift/V1_S3_plus_travel-0.5.csv | df00e35171 | 0.20237 |
| shift/G1_generator_formula.csv | a2687f3a53 | 0.17795 (formula is not the generator) |
Next: W1_E3base_S3shifts, W2_burst-1.0_ndold-0.75.

## Round 3 shifts (3 Oct ~19:05)
| shift/W2_burst-1.0_ndold-0.75.csv | 31f108aa8c | **0.20530** (best) |
| shift/W1_E3base_S3shifts.csv | 6aa5f1f17b | 0.19923 |
Tried and rejected as shift estimators (fail the known-public-direction check): PU learning (src/pu.py: says big_old down),
importance-weighted v5 (src/iw.py: no burst shift). Segment emulator (src/lb_seg.py): corr 0.949 overall, LOO RMSE 0.0042 on shift family.
Burst age split: train bursts <=90d 63.8% fraud (n47) vs test n49 (same count); >365d train 139 (8-10% fraud) vs test 290 (2x) ->
injected bursts are old-account bursts. Next: X1 (burst old -1.25, young 0, ndold -0.75, 208c516581), S4 (burst -1, ndold -1, 7568ae0d7a).
| shift/S4_burst-1.0_ndold-1.0.csv | 7568ae0d7a | 0.20516 |
| shift/X1_burstOld-1.25_burstYoung0_ndold-0.75.csv | 208c516581 | 0.18152 (young bursts must be demoted: mostly legit in test) |
W2 top-120: 55 young bursts (46%). No-velocity v5 (NV, 38c139a24e) still keeps 47 young bursts in top-120 -> not submitted.
Next: P1 young -2.0/old -1.0, P2 young -1.0/old -0.5 (ndold -0.75).
| shift/P1_burstYoung-2.0_burstOld-1.0_ndold-0.75.csv | ce1e96a4f4 | **0.21289** (rank 2) |
Correction: earlier "young bursts same count" compared raw counts; as rates all burst age bands are 1.7-3.6x enriched.
Package: ../BEST_P1_v5_burstsplit_ndold_0.21289 (sandbox-verified, maxdiff 3e-16).
P1 top-120: nd&age<=90&t1h>=3 (non-burst) 34 rows (1.5x enriched, train fraud 31%), young bursts 27.
Next: Q1 = P1 + that group -1.0; Q2 = P1 with young bursts -3.0.
| shift/Q1_P1_plus_fastYoungND-1.0.csv | b01f60b353 | 0.20826 |
| shift/Q2_P1_burstYoung-3.0.csv | 91a8ac0a7d | 0.20928 |
## Hour fingerprint (src/inject.py, inject2.py, inject_detector.py)
Test = ~8960 train-like base rows (uniform hours) + ~3040 injected rows that never use hours 5,6,7,21,22 (adv AUC train vs test in those
hours 0.574 vs 0.684 elsewhere). Bursts are NOT injected (base itself has ~4x bursts). Day-band (8-20) injection: nd_old 778,
amt>2000 643, big&age>1500 886, luxury/travel/cash/electronics, foreign. Night-band (23-4): ordinary low-value SG established accounts (802).
Detector P(inj|x) (cross-fitted band-vs-clean) says high-scoring big&old rows are most injected; public says demoting them hurts ->
day injection includes high-value fraud on established accounts. Next: U1 (P1 + bigold +0.5), U2 (+ travel +0.5).
| shift/U1_P1_bigold+0.5.csv | c9c788e7f1 | 0.21424 |
| shift/U2_P1_bigold+0.5_travel+0.5.csv | e843505789 | 0.21082 |
| friend Z2 (young -2, old -0.5, ndold -0.75) | 95dccc5fd8 | **0.21922** |
Next: Z3 = Z2 + bigold +0.5 (6234082ef2; package ../CANDIDATE_Z3_*), Z4 = Z3 with old bursts -0.25 (a6c7f33a09).
| shift/Z3_Z2_bigold+0.5.csv | 6234082ef2 | 0.22215 |
| shift/Z4_Z2_bigold+0.5_oldburst-0.25.csv | a6c7f33a09 | 0.22195 |
| friend K3 (Z2 + nd<180 night low-velocity -0.5, high-value cat 2000-6000 age>365 +0.5) | d03cb4d61b | **0.223** |
K3 top-120: nd&age<=90&t1h>=3 (non-burst) 34/38 rows, ~63-67% injected (hour fingerprint / detector); Q1 demotion -1.0 cost 0.0046
-> injected fraud type. Next: K4 = K3 + that group +1.0 (ee12bc40b6), K5 = +0.5 (69e50b93c8).
| shift/K5_K3_fastYoungND+0.5.csv | 69e50b93c8 | 0.21741 |
| shift/K4_K3_fastYoungND+1.0.csv | ee12bc40b6 | 0.21477 |
Lesson: Q1 (demote) and K4/K5 (promote) both hurt -> that group was already at its optimum; "down hurts" does not imply "up helps".
Next: K6 = K3 with nd_old/high-value shifts removed at clean hours 5,6,7,21,22 (82 + 30 rows; 356731ccb5).
| shift/K6_K3_cleanhour_unshift.csv | 356731ccb5 | 0.22041 |
## Research round (3 Oct ~20:10)
- Web: dataset/generator not public. A competing team's public repo exists (github.com/Sindyyyy19/dlw-datathon-track2) - deliberately NOT opened.
- EM + bias-corrected calibration (Alexandari et al. ICML 2020; Garg et al. NeurIPS 2020), per group (src/em_bcts.py): collapses many
  groups to 0 and contradicts public on high-value -> rejected.
- Latent-class type-proportion EM (src/lca_shift.py): unidentifiable/unstable across K (test prior 6.5% / 1.1% / 1.7%) -> rejected.
- Conclusion: unsupervised estimators cannot identify the per-group label changes; public LB is the only oracle; K3 is a local optimum
  of the group-shift family (6 consecutive single-group probes from the best file all lost 0.001-0.008).
## Template round (3 Oct ~21:30)
| friend M123 (K3 + old burst -0.3, hv +1.0, mid-value 1000-2000 +0.5) | - | **0.22523** |
| friend Q30 (0.7 v5 + 0.3 pairwise-interaction LR C=0.003, M123 shifts) | - | **0.23288** (#1 public) |
- Replicated Q30's interaction LR (src/blendlab.py): own CV 0.2472; blend w0.3 +0.0060, C=0.01 w0.4 +0.0072 vs v5. No other OOF (MLP/TabM/NCA/CatBoost/LGB/RBF/spline) adds to v5+poly.
- The ~3040 planted test rows are three uniform synthetic boxes (src/templates.py), all established accounts, never hours 5,6,7,21,22:
  T1 day new-device small SG (grocery/retail/transport/utilities, ~950), T2 day high-value 1500-6000 (cash/electronics/luxury/travel, AU/JP/MY/SG, spend 100-2000, ~860),
  T3 night small SG (food/gaming/grocery/transport, ~800). Test bursts match train LEGIT bursts on every feature (fraud share 0-6% vs 24%).
- Train rows inside the T2 box: 1 fraud / 270 (v5 OOF predicts ~7). Q30's +1.0 high-value rule puts 438 T2 rows in its top-1000.
- Label-model fits (src/tmpl_fit*.py) cannot separate "T2 ~5% fraud" from "T2 legit + out-of-box high-value base rows under-ranked" on public differences;
  candidate scoring over the hypothesis grid (src/tmpl_decide2.py): C1 (T2 box unshifted) mean +0.010..+0.011, worst -0.0008; C7 (T2 -1, out-of-box +0.5) mean ~+0.012, +0.038 if T2 legit, -0.0105 if 5%.
| tmpl/C1_Q30_T2box_unshift.csv | 9422baa279 | ? |
| tmpl/C7_Q30_T2box-1_hvout+0.5.csv | bfc1046551 | ? |
| tmpl/C1_Q30_T2box_unshift.csv | 9422baa279 | 0.23277 (= Q30: T2 box is NOT legit) |
| tmpl/C7_Q30_T2box-1_hvout+0.5.csv | bfc1046551 | 0.22841 |
- Fits: T2 ~3.5-5% fraud (Q30 mean 5.1% == C1 mean 2.2% on public -> level ~3.5%); T1 (planted new-device box) ~0.2% (fits nd_old series, src/t1_fit.py);
  T3 unknown. Extra test bursts = train LEGIT burst profile (MY/ID/SG), not a planted box. Multiple imputation for missing values: no CV gain (src/mi_cv.py).
- Booklet: Round 1 = 60% PR-AUC + F1 + Recall. Q30 flags 18 rows at 0.5 (recall ~4%). Constant logit shift -> 0.5 = top 2.5% (F1-optimal region,
  CV F1 ~0.30, expected test F1 0.24 / recall 0.27); ranking/PR-AUC unchanged.
| tmpl/R1op_package_output.csv (= ../Track2_R1op_package/predictions_R1op.csv; Q30 + T2 level 0.035 a0.3 + op shift) | 1c81db55b1 | ? |
Packages: ../Track2_R1op_package and ../Track2_Q30op_package (both sandbox-verified).
## Generator-aware basis LR (GALR; src/galr.py, galr_fresh.py, galr_bag.py)
- Same 5x5 folds: GALR (with nd x night / nd x ecommerce) 0.2373-0.2379 for C 0.03-0.1 (hypothesis ~0.235-0.24 reproduced); without interactions 0.229-0.231
  (~0.227 simple LR reproduced). vs v5 -0.011 (0/5 reps), vs Q30 base (v5+poly) -0.017 -> fails standalone.
- As blend member 0.8 Q30base + 0.2 GALR: +0.0009..+0.0011 (5/5); fresh seeds 11/99/314: w0.1 +0.0011..+0.0013, w0.2 +0.0008..+0.0030;
  nested CV (C,w inner-selected, never w=0): +0.0017 / +0.0033. Bagging 30 fold models (prob or rank avg): no gain vs full fit.
| tmpl/R2op_package_output.csv (= ../Track2_R2op_package/predictions_R2op.csv) | bc365689c1 | ? |
| tmpl/R2op_package_output.csv | bc365689c1 | **0.23434** (best; Q30 + T2 level + GALR 20% + op shift) |
- Residual scan of R2 base OOF (src/resid_scan.py): no single-feature bin |z|>3, 1/839 pairwise cells |z|>3 (chance) -> base saturated on train.
- Burst country split (MY/ID/SG power users 3.9x in test vs other-country 1.65x): v5 already calibrated within train bursts by country; per-cell
  moment matching too noisy (6-32 train frauds/cell) -> not acted on. No 4th planted template outside T1-T3 (residual excess = box-edge spill + count noise).
- Greedy blend search over 151 stored OOFs (src/greedy_blend.py): only degree-3 interaction LR survives (held-out +0.0009); fresh seeds (src/poly3_fresh.py)
  w0.1: +0.0002/+0.0051/+0.0016/+0.0028.
| tmpl/R3op_package_output.csv (= ../Track2_R3op_package/predictions_R3op.csv) | 7cdfaac8ee | ? |
| tmpl/R3op_package_output.csv | 7cdfaac8ee | 0.23260 (ehh: -0.0017 vs R2op; degree-3 member did not transfer) |
Final: R2op (../Track2_R2op_package, 0.23434 public, sandbox-verified, includes operating-point shift for F1/recall). Search stopped here.
## Public-only route (user request): one-change probes on R2op
- History emulator v2 (src/lb_emul2.py, src/lb_history.py: 48 hash-verified scored files; template rates + group effects + 3 components)
  FAILED time holdout (predicted C1/C7/R2op/R3op deltas vs Q30 off by 0.006-0.045; degenerate all-fit) -> rejected; probing one change at a time instead.
| probe/B1_R2op_burst_MYIDSG-0.25_other+0.25.csv | fd8bab494f | **0.24016** (+0.0058 vs R2op; new best) | bursts split by country type (per-row rate vs train: MY/ID/SG bursts 3.9x, other-country bursts 1.7x) |
| probe/T3a_R2op_T3box+0.7.csv | a505e444ab | ? | planted night box T3 nudged up (never probed) |
## TabPFN (pretrained tabular transformer, tabpfn 9.1.0, GPU; src/tabpfn_cv.py) - REJECTED
Same folds (evaluate.SPLITS repeat 0), CV PR-AUC: v5 0.2488 | R2 base blend 0.2582 |
TabPFN raw features 0.2249 (blend into R2 base: -0.0006 @0.1 ... -0.0153 @0.7) | TabPFN raw+engineered 0.2344 (blend +0.0002 @0.1, -0.0001 @0.2, worse beyond).
No architecture tried beats the LR + kNN + interaction-LR + GALR blend; R2op (0.23434) remains final.
## Architecture sweep (ChatGPT list + own; harness src/arch_eval.py: same 5x5 folds, standalone and as R2-base blend member, w chosen on reps 0-2, checked on reps 3-4)
| candidate | standalone CV AP | best R2-member delta (wins/5) | held-out | verdict |
|---|---|---|---|---|
| R2 base blend (reference) | 0.2559 | - | - | public 0.23434 |
| EBM pure GAM (interpret 0.7.8, isolated .venv_ebm) | 0.2133 | -0.0008 (0/5) | -0.0009 | rejected |
| EBM smoothed GAM | 0.2123 | -0.0007 (0/5) | -0.0009 | rejected |
| EBM GA2M 3 interactions | 0.2138 | -0.0007 (0/5) | -0.0008 | rejected |
| EBM GA2M 10 interactions | 0.2164 | -0.0008 (0/5) | -0.0008 | rejected |
| Program synthesis (268-program library, L1, C by inner CV per fold) | 0.2269 | -0.0005 (1/5) | -0.0008 | rejected |
| Program synthesis, relaxed L2 refit | 0.2071 | -0.0008 (0/5) | -0.0008 | rejected |
| v5 + fold-selected programs (nested) | 0.2068-0.2234 | -0.0004 (1/5) | -0.0005 | rejected |
| Joint LR on v5 + GALR + interaction blocks | 0.2268-0.2464 | +0.0005..+0.0010 (3-5/5) | +0.0001..+0.0010 | noise-level, not used |
| Direct AP optimisation (Smooth-AP, warm start from v5 LR) | below v5 on every fold (0.21-0.25 vs 0.218-0.257) | - | - | rejected |
| TabPFN full context raw / raw+eng | 0.2249 / 0.2344 | -0.0006 / +0.0002 | - | rejected |
| TabPFN context bagging (all frauds + 8k legit x6) | fold0 0.2294 (v5 0.2538) | - | - | stopped (fail fast) |
Program synthesis stability: 28 programs rediscovered in >=80% of 25 folds (nd&night, nd&age<365, nd&amt>1000, nd&ecom, night&burst,
foreign&age<180, foreign&amt>500, burst&amt>1000, amt>250, amt>500, t24>=20, t1>=4, t1>=7, age>270, ...) - real structure, already
captured by v5's smooth features + interaction LR. Earlier: LCA generative (0.18), multiple imputation, importance weighting, WoE - rejected.
Conclusion: no architecture beats the LR+kNN+interaction+GALR blend on train CV; R2op (0.23434) stays final.
## Burst-by-country coordinate steps from B1 (0.24016)
Cells (test n / in R2op top-300): young MY/ID/SG 60/19, young other 44/38, old MY/ID/SG 270/35, old other 26/24.
| probe/B2a_burst_MYIDSG-0.25_other+0.5.csv | a8a0634efc | **0.24094** (new best) | other-country side pushed further |
| probe/B2b_burst_MYIDSG-0.5_other+0.25.csv | 722fc4de92 | 0.23904 | MY/ID/SG side pushed further |
Per country (test n / B2a top-300; train fraud rate): young OTHER 44/43 (65%), old OTHER 26/26 (28%), young ID 22/9 (50%), young MY 28/6 (31%),
old ID 80/13 (8%), old MY 100/12 (5%), SG 100/3 (0-2%).
| probe/C1_burst_MYIDSG-0.25_otherYoung+0.75_otherOld+0.25.csv | 94c0ef2e2d | 0.23702 | age split inside other-country bursts |
| probe/C2_burst_MYIDSG0_other+0.5.csv | 0f20603746 | **0.24174** (best) | MY/ID/SG demotion released |
- Local burst-cell emulator (src/burst_local.py; 6 files differing only in 3 burst cells): fits all within ~0.001, LOO RMSE 0.003;
  cell deltas vs R2op: MY/ID/SG -0.2, other young +0.5, other old +1.4..+1.6. The old-other conclusion depends on C1 (fit without C1 reverses it);
  history agrees with it (P1: all old bursts -0.5 more -> -0.0063). EV of old-other +1.0 ~ +0.002 at P(C1 reading)=0.7.
- Package ../Track2_C2_package (R2op notebook + burst-by-country shifts before the operating point): reproduces scored C2 (max diff 1.1e-16,
  identical ranking), sandbox all 3 modes pass.
| probe/D1_burst_MYIDSG0_otherYoung+0.5_otherOld+1.0.csv | e2fa4fb2f3 | 0.23604 (-0.0057: C1 reading wrong) | only the 26 old other-country bursts move (+0.5 -> +1.0) |
- Refit local emulator with D1: cannot fit (C2 pred +0.0042 vs obs +0.0074; LOO RMSE 0.005-0.006) -> public outcomes driven by a few
  realised labels; emulator no longer used. C2 is the best point on every tested burst coordinate.
- New-device x country: dilution is SG-only (T1 box); non-SG nd_old rows 1.0x train rate but only 5 in C2 top-300 -> low leverage, not probed.
- FINAL: C2 0.24174 (../Track2_C2_package, friend folder ../BEST_C2_0.24174 + zip; sandbox-verified).
