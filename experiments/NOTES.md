# Track 2 experiment log (concise)

## Rules (from PDFs/template)
- Metric PR-AUC (track page). Booklet: Round 1 = 60% "PR-AUC, F1, Recall" + 40% understanding. Deadline 11:30 AM 4 Oct.
- Platform test: no id column, rows shuffled -> features must be row-local. Output: `prediction` column, same row order.
- Private eval re-runs our notebook on hidden data that "may contain different observations and edge cases".
- Sandbox: no internet, image libs in requirements-image.txt (no catboost). 1 package upload / 2h.

## Data facts
- train 20000 x 12, test 12000; fraud 353 (1.765%). No dups, no train/test overlap, id uninformative, no entity/time keys.
- Missing ~0.6%/col train vs ~2.2%/col test (MCAR-looking in train; missing indicators are noise -> dropped).
- Adversarial AUC 0.667: test has heavier amount tail, older accounts, 3-5x more velocity bursts (tx24>=16), more new_device.
- Signals: tx_1h>=3 (step up, ~20% fraud at >=5), tx24 burst cluster >=16 (~25%), account_age<90 (~4.4%),
  new_device (7.5% vs 1.2%), high amount within cash_transfer/electronics/luxury, hour (night higher, hour as category helps).

## Validation
- 5x5 RepeatedStratifiedKFold (no groups/time exist). Also "test-miss": inject 1.7%/col extra MCAR missingness into val folds.

## Results (pr_auc mean over 5 repeats | test-miss)
- lr_raw_log 0.210 | lgb_default 0.171 | lgb_reg 0.208 | hgb_reg 0.198
- lr_eng (hourcat, eng feats, indicators) 0.227 | 0.213
- lr_eng_noind C=0.1 0.2286 | 0.2225   <- v1
- splines 0.200 (overfit); interactions 0.226; missing-aug 0.225; lgb_eng_mono 0.212 | 0.206
- blend 0.7 LR + 0.3 lgb_mono (prob avg) 0.231

## Round 3 (hardening, shift-aware validation, ablation)
- v1 CRASHED on edge input (tx24=0 -> inf). v2 = same model + sanitize(): missing cols->NaN, to_numeric, clip to train range, normalise category strings, inf->NaN. Identical preds on test.
- Adversarial density-ratio weights (adv AUC 0.669, ESS 9.7k): implied test fraud prevalence ~2.2%; test-weighted PR-AUC ~0.34. Model ranking identical under weighted/unweighted/test-miss views.
- Additive GBMs: stumps 0.207, depth2 0.210, depth2 mono 0.212, xgb depth2 0.208; LR with binned velocity/age 0.223. CatBoost d4 0.219 / d6 0.219 (3x5).
- Ablation (drop group -> PR-AUC; full 0.2286): new_device 0.188, age 0.207, amount 0.215, velocity_24h 0.217, night 0.223, merchant 0.222, channel 0.225, hour-cat 0.227, spend/foreign/country ~0. class_weight=balanced 0.191 (hurts!).
- Dropping linear 1h velocity helps on 2 CV seeds (+0.002-0.003; tw +0.01): within the burst cluster (tx24>=16, 85% have tx1h>=5) tx1h adds nothing. Kept a non-burst tx1h>=4 flag.
- v3 = LR C=0.1, v2 features minus 1h terms plus t1_ge4_nb: 0.2303 | tw 0.349 | test-miss 0.2257.
- Blends with v3 (3 reps): +CatBoost 0.3 -> 0.2344 (tw 0.358); +lgb_eng_mono 0.3 -> 0.2321; within noise; extra dependency not justified.
- Error analysis v3: calibrated at natural prior (decile pred~obs; brier .0152 vs prior .0173). Top0.5%: P .56 R .16; top2%: P .27 R .31; best F1 .307 @ thr .147 (1.1% flagged).
  Signature fraud (burst AP .71, new_device AP .51, age<90 AP .45) vs camouflaged fraud (17% of frauds below median rank: small food/transport/grocery, old accounts, no new device, SG). Trees are worse than LR in hard segments too -> noise floor.
