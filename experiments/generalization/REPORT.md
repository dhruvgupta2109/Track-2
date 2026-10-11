# Train-only candidate for Track 2

This is a separate candidate. The scored C2 final submission is unchanged. This candidate has no public submission hash or measured public/private score.

## What changed

- Removed all public-leaderboard-tuned group shifts, high-value look-alike rules and country-burst corrections.
- Retrained the same three model families with category amount medians from training alone. No public-test features are read during fitting.
- Kept the 56% v5 / 24% interaction / 20% GALR logit blend and their existing regularisation settings.
- Kept nearest-neighbour features: removing them slightly reduced local average precision.
- Estimated a decision threshold from training calibration splits; final threshold **0.146116**. A fixed logit shift maps it to 0.5. This preserves ranking, but outputs are decision scores rather than calibrated probabilities.
- Added empty-input and nested-output-directory handling.

## What the experiment shows

Five stratified outer folds for each of seeds 43017 and 92041. Each outer validation row is unseen by its fitted model; all preprocessing is fitted within the training fold for the candidate. The original transductive recipe uses outer-training plus unlabeled validation category medians as a control. The actual C2 corrections are copied from the final notebook. Their full-data implementation reproduces the scored C2 CSV within 3.61e-16.

The rows and labels have been used in extensive earlier development. Different seeds reuse the same 353 fraud labels. These are local diagnostic estimates, not an untouched test set or statistically independent experiments. All candidate settings were fixed before this run; no hyperparameter sweep or new public submission was performed.

Mean average precision (the repository's PR-AUC metric):

| view | c2_recipe | unadjusted_transductive | train_only | train_only_no_knn |
| --- | --- | --- | --- | --- |
| standard | 0.2159 | 0.2475 | 0.2475 | 0.2449 |
| missing_5pct | 0.2043 | 0.2308 | 0.2309 | 0.2285 |
| missing_10pct | 0.1962 | 0.2323 | 0.2322 | 0.2319 |
| bursts_x3 | 0.3311 | 0.3750 | 0.3749 | 0.3713 |
| bursts_x0.3 | 0.1584 | 0.1792 | 0.1792 | 0.1771 |
| older_accounts_x3 | 0.1344 | 0.1601 | 0.1601 | 0.1585 |
| new_devices_x3 | 0.3369 | 0.3778 | 0.3778 | 0.3735 |
| non_SG_x3 | 0.2923 | 0.3340 | 0.3340 | 0.3310 |
| fraud_prior_0.005 | 0.1030 | 0.1389 | 0.1388 | 0.1359 |
| fraud_prior_0.01 | 0.1591 | 0.1931 | 0.1931 | 0.1903 |

Missingness masks an extra 5% or 10% of each feature in validation only. Other population views reweight existing labeled rows. Population views assume the conditional fraud mechanism remains unchanged; their absolute AP values are not directly comparable across different prevalences. Compare models within each row.

The main improvement comes from removing public corrections, not replacing test-derived medians: the unadjusted control and train-only candidate are nearly identical. A lower train-CV score from C2 is not proof of leaderboard overfitting, because its rules were intended for a different distribution.

Threshold results on outer validation:

| model | ap | f1 | recall |
| --- | --- | --- | --- |
| c2_recipe | 0.2159 | 0.2870 | 0.2720 |
| train_only | 0.2475 | 0.2937 | 0.2720 |

For the candidate, every outer threshold is chosen on a separate 25% split of outer training, using another model fit on the remaining 75%. The final threshold is the median of ten thresholds. Refitting on all training data can change score calibration. There is no proof this threshold is optimal on private data. C2 uses its original fixed shift; control-model F1 results in the CSV use the candidate threshold diagnostically and are not separately tuned.

## If the conditional fraud rate changes

After fixing the candidate, a diagnostic reweights only legitimate examples in specified groups by 3x or 10x. This represents one possible dilution of risk, not private-label knowledge. No model choice or parameter is changed based on this check.

| view | c2_recipe | train_only |
| --- | --- | --- |
| legit_bursts_x10 | 0.1349 | 0.1251 |
| legit_bursts_x3 | 0.1846 | 0.1953 |
| legit_high_value_older_x10 | 0.1357 | 0.2139 |
| legit_high_value_older_x3 | 0.1872 | 0.2373 |
| legit_older_new_device_x10 | 0.1744 | 0.1891 |
| legit_older_new_device_x3 | 0.2037 | 0.2256 |

These checks only amplify observed legitimate examples. They do not generate unseen fraud mechanisms or the public set's suspected synthetic look-alikes. Neither model is guaranteed robust to arbitrary concept shift. C2 has higher AP when legitimate bursts are multiplied tenfold (0.1349 vs 0.1251). It also has better F1 when legitimate older-account new-device rows are multiplied 3x or 10x, even though the candidate has higher AP. Public corrections may help if their patterns recur in private data; removing them can sacrifice that advantage. This candidate is less dependent on public feedback, not universally more robust.

Country holdouts exclude the listed countries from all fitting and evaluate on them as unseen categories:

| held_countries | model | rows | positives | ap |
| --- | --- | --- | --- | --- |
| AU/JP | train_only | 966 | 30 | 0.3723 |
| AU/JP | train_only_no_knn | 966 | 30 | 0.3834 |
| MY/ID | train_only | 2065 | 52 | 0.3337 |
| MY/ID | train_only_no_knn | 2065 | 52 | 0.3338 |

## Files and validation

Upload `prediction_notebook.ipynb` with **this folder's** `model.pkl` if choosing this candidate. Do not mix it with C2's model. No requirements file is needed beyond the preinstalled numpy, pandas, scikit-learn and joblib libraries. `reference_predictions_unscored.csv` is a local reference and should not be uploaded as part of the notebook package.

Six isolated notebook executions passed with the organiser-version Python environment: full input, shuffled input without id, malformed/missing columns, all-missing rows, header-only input and zero-byte input. Output is one `prediction` column, finite values in [0,1], and input row order. Singleton and batch predictions agree. Candidate flags 230 of 12,000 local test rows at 0.5; C2 flags 307. The flagged rate is diagnostic, not a target inferred from hidden labels.

No Linux-platform run, new public evaluation or private evaluation was performed. C2 file hashes were checked before and after all work and are unchanged.

Evidence: `Track 2/experiments/generalization/` contains protocol, OOF caches, detailed metrics, thresholds, country holdouts, baseline equivalence and notebook validation. Reproduce from `Track 2` using `.venv_clean/Scripts/python.exe src/generalization_check.py` then `src/generalization_validate.py` with the same interpreter.

Validation follows scikit-learn guidance on [keeping preprocessing within training folds](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) and [separating classifier fitting from threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html).
