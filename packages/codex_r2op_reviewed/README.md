# R2op reviewed delivery

This is a separate Codex-branch delivery of the recorded **0.23434 public PR-AUC** winner. It does not claim a new public result. The trained model is byte-identical to the original R2op bundle. The revised notebook preserves every score to floating-point precision, handles empty input, creates the requested output directory, and predicts in bounded chunks.

Upload these four files individually to their matching portal cards:

| Card | File |
|---|---|
| Prediction notebook | `prediction_notebook.ipynb` |
| Trained model | `model.pkl` |
| Requirements | `requirements.txt` |
| One-page technical report | `technical_report.pdf` |

`predictions_R2op.csv` is the equivalent public-test prediction file. Keep the existing R2op final selection unless a new scored result justifies changing it. `predictions_before_operating_shift.csv` is an explicitly separate reference; it preserves the same ranking while omitting the final +2.188389 logit shift. It still includes public-tuned group rules and is not guaranteed calibrated. Do not confuse these files or upload the judge analysis notebook to the prediction card.

The input path is `DATATHON_INPUT_PATH`, default `test.csv`; the output path is `DATATHON_OUTPUT_PATH`, default `predictions.csv`. ID-less input outputs only `prediction`; input with an ID preserves its ID and order. Predictions use fixed training/reference constants. The 2.5% public workload is **not** recomputed on new batches.

The final operating scores are in [0,1], but they are not calibrated probabilities. The +2.188389 shift places the 0.5 threshold at a pre-shift score of approximately 0.100798 and flags 300 public-test rows. The evaluator's private threshold is unspecified, so no improved private F1 or recall is claimed.

The source repository contains the executed `analysis/Track2_Judge_Analysis.ipynb`, which provides data analysis, full training reproduction, fresh CV and judge Q&A. It is a separate local showcase; it needs the supplied CSVs and repository source. It does not need ignored NumPy caches because it regenerates them.

Validation: eight clean-environment notebook cases; singleton/batch independence; duplicate, reordered-column and category-normalisation checks. Fresh training reproduces the original scored outputs within 2.8e-12. The model SHA-256 is `60d9e77b6234ee01af848c2dfc95b5fb37b15d38fc845010422dc96b3f36add7`. See `manifest.json` and the repository's `experiments/codex/package_validation.json` for evidence.

The archive at `deliverables/Track2_R2op_reviewed.zip` contains this delivery folder. Unzip it to access the individual upload files. No portal upload, private evaluation, or Git push has been performed by this branch.
