# Codex review and delivery improvement

Branch: `codex/track2-evidence-improvements`, based on `4eae1c8`.

## Decision

Retain the recorded public winner R2op (0.23434). No new predictive winner was established and no public submissions were uploaded. Three focused interaction-member hypotheses were measured on fresh folds, with a confirmation gate written into the experiment before evaluation. None passed.

| Variant | Confirmation mean AP change | Extra-missingness mean AP change | Decision |
|---|---:|---:|---|
| Leave-one-out category fraud-risk features | +0.000374 | +0.000504 | Below gate |
| Degree-two interactions without squared terms | +0.000491 | +0.000524 | Below gate |
| Cap continuous interaction inputs at 3 training SD | -0.001412 | -0.001312 | Worse |

Screen: seed 2718. Confirmation: seeds 811 and 1201. Five folds per seed, C=0.003 and final blend weights held fixed. The gate required a mean confirmation AP gain >=0.001, gains on both standard confirmation seeds, and a mean extra-missingness loss no worse than -0.001. This is a practical stopping rule, not a significance test. Different folds reuse the same rows and labels.

For interaction-only, a paired stratified row bootstrap of the confirmation mean OOF predictions gives a 95% interval of approximately -0.00023 to +0.00265. It conditions on fitted predictions and excludes retraining uncertainty. The mean-OOF AP delta (+0.000889) is a different calculation from averaging per-seed AP deltas (+0.000491). It does not establish a reliable test gain.

## Training-recipe discrepancy found and resolved

The first fresh full-training attempt differed from R2op by a maximum score difference of 0.0050878. v5 and GALR reproduced, but the interaction member did not. Inspecting the saved constants showed that its category amount feature uses **log1p(median(raw amount)) from train plus unlabeled public-test features**. The earlier local replication used median(log amount) from training alone.

The combined raw medians match every saved category constant exactly. Incorporating this detail reproduces the complete scored model from fresh full training to within 2.8e-12. The initial inductive screen CSVs are retained under `experiments/codex/initial_inductive/`. The final experiment was rerun using train plus unlabeled outer-validation medians for all compared variants. Fraud-risk maps and nearest-neighbour references always use outer-training labels only. This is transductive validation, explicitly described in the report/notebook.

## Delivery changes

- The original sibling package and public winner were preserved. Model bytes are unchanged.
- The row-local numerical computation is exposed in `src/r2_runtime.py` and embedded directly in the revised prediction notebook; no repository import is required in the sandbox.
- The prediction driver handles empty/header-only files, creates nested output directories and uses fixed-size chunks. Constants and thresholds stay fixed across input batches.
- Eight isolated notebook cases pass using the existing organiser-version clean environment. The validator raises on failure; unlike the earlier sandbox script, failure cannot silently exit successfully.
- A standalone, executed judge notebook provides EDA, feature examples, fresh full training, fresh CV, workload curves, recorded public history and Q&A. It is separate from the prediction-only upload.
- The required one-page report follows the organiser's five-section example from `../Datathon Track 2 - suraj try/report_format.pdf`. It was rendered and visually checked.

## Corrections to the inherited summary

The actual burst **rate ratio is 3.118x**, rather than 4x. Mean feature missingness is **0.519% train / 1.811% test**; individual columns vary. Clean-hour extrapolation estimates about 3,038 excess rows under its assumptions. The feature boxes contain 1,017 T1 / 975 T2 / 1,040 T3 test rows, including base-like rows; they are not exact planted-row counts.

T2's 3.5% level and random labels within synthetic-looking boxes are hypotheses inferred from aggregate public probes, not observed label facts. The original residual scan does not prove an attainable PR-AUC ceiling. No current leaderboard rank was checked.

The final R2op pre-operating scores flag **16** public-test rows at 0.5 (the earlier 18 referred to Q30). The fixed operating shift flags 300 rows and preserves ranking, but changes calibration. No supplied specification states the private evaluator's threshold. Expected private F1/recall numbers are consequently omitted from the report. The pre-shift CSV is provided as a reference and is also not guaranteed calibrated because public-tuned adjustments remain.

## Reproduce in this repository

Use Python with the pinned organiser libraries already listed in `requirements-image.txt`. The PDF authoring script additionally uses reportlab and PyMuPDF; these are authoring-only dependencies and are excluded from submission requirements.

```powershell
.venv\Scripts\python.exe src\codex_delivery.py
.venv\Scripts\python.exe src\codex_polylab.py --seeds 2718 811 1201
.venv\Scripts\python.exe src\codex_evidence.py
.venv\Scripts\python.exe src\codex_validate.py
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --output-dir tmp analysis\Track2_Judge_Analysis.ipynb
```

The delivery builder can use the committed original-notebook snapshot if the sibling original package is unavailable. The pinned trained model remains in the reviewed package, and the original scored CSV remains in `submissions/tmpl/R2op_package_output.csv`. Generated NumPy caches are ignored; the judge notebook recreates them.

Not performed: public upload/evaluation, private evaluation, proof of private F1/recall, verification of the private threshold, current leaderboard-rank verification, GPU training, or Git push. The pre-existing untracked `experiments/blendlab_r1.txt` was not changed or staged.
