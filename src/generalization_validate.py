"""Verify the standalone candidate and document limits of the shift evidence."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from generalization_check import ROOT, C2, OUT, PACKAGE, SEEDS, NAMES, sha, record, predict_members, notebook_runtime, adjustments


def main():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["C2_hashes"] == {p.name: sha(p) for p in C2.iterdir() if p.is_file()}
    tr = pd.read_csv(ROOT / "train.csv")
    te = pd.read_csv(ROOT / "test.csv")
    y = tr.fraud.to_numpy()
    # These class-conditional reweightings change P(y|x). They are synthetic
    # sensitivity checks, not inferred private labels or an optimization target.
    groups = {"legit_bursts": tr.transactions_last_24h.ge(16).to_numpy() & (y == 0),
              "legit_older_new_device": (tr.account_age.gt(365) & tr.new_device.eq(1)).to_numpy() & (y == 0),
              "legit_high_value_older": (tr.account_age.gt(365) & tr.transaction_amount.between(1500,6000)
                                        & tr.merchant_category.isin(["luxury","cash_transfer","electronics","travel"])).to_numpy() & (y == 0)}
    concept_protocol = {"purpose": "Diagnostic only; candidate, hyperparameters and threshold fixed before this check.",
                        "multipliers": [3, 10], "groups": list(groups),
                        "limitations": "Only reweights observed legitimate examples; cannot simulate genuinely unseen fraud mechanisms or synthetic look-alikes. No private labels used."}
    (OUT / "concept_protocol.json").write_text(json.dumps(concept_protocol, indent=2), encoding="utf-8")
    rows = []
    for seed in SEEDS:
        cache = np.load(OUT / f"oof_{seed}.npz")
        for group, mask in groups.items():
            for multiplier in [3,10]:
                w = np.where(mask, multiplier, 1.)
                for name in NAMES:
                    p = cache[f"standard__{name}"]
                    decision = p >= (.5 if name == "c2_recipe" else cache["threshold"])
                    rows.append(record(seed, f"{group}_x{multiplier}", name, y, p, w, decision))
    pd.DataFrame(rows).to_csv(OUT / "concept_metrics.csv", index=False)
    concept = pd.DataFrame(rows).groupby(["view","model"])[["ap","f1"]].mean().reset_index()
    concept.to_csv(OUT / "concept_summary.csv", index=False)

    b = joblib.load(PACKAGE / "model.pkl")
    members = (b, b["poly_model"], b["poly_constants"], b["galr_model"])
    def expected(frame):
        if not len(frame):
            return np.empty(0)
        return expit(logit(np.clip(predict_members(members, frame), 1e-9, 1-1e-9)) - logit(b["decision_threshold"]))
    with threadpool_limits(limits=2):
        full = expected(te)
        # Verify that the copied adjustment rules exactly reproduce the saved C2 output.
        old = joblib.load(C2 / "model.pkl")
        om = (old, old["poly_model"], old["poly_constants"], old["galr_model"])
        _, runtime, rules = notebook_runtime()
        c2 = adjustments(te, old, predict_members(om, te), runtime, rules)
        ref = pd.read_csv(C2 / "reference_C2_public_0.24174.csv")
        if "id" in ref:
            ref = ref.set_index("id").loc[te.id].reset_index()
        c2_difference = float(np.max(np.abs(c2 - ref.prediction.to_numpy())))
        assert c2_difference < 1e-9, c2_difference
        rng = np.random.default_rng(73)
        indices = rng.choice(len(te), 25, replace=False)
        one = np.array([expected(te.iloc[[i]])[0] for i in indices])
        assert np.max(np.abs(one - full[indices])) < 1e-9
        # Whole-set reference uses the actual inference functions, independent of the driver.
        pd.DataFrame({"prediction": full}).to_csv(PACKAGE / "reference_predictions_unscored.csv", index=False)

    edge = pd.DataFrame({"transaction_amount": ["bad", -5, np.inf, 1e9, 0],
                         "transaction_hour": ["bad", -1, 25, np.nan, 12],
                         "country": ["XX", None, " sg ", "AU", "JP"],
                         "new_device": [2, "yes", np.nan, 0, 1],
                         "transactions_last_24h": [0, -1, 10000, np.nan, 2]})
    shuffled = te.sample(frac=1, random_state=53).drop(columns="id")
    cases = [("full_default_paths", te, False), ("shuffled_no_id_env_paths", shuffled, True),
             ("malformed_missing_columns", edge, True), ("all_missing", pd.DataFrame({"country": [None]*3}), True),
             ("empty_header", te.iloc[:0], True), ("zero_byte_input", None, True)]
    checks = []
    python = ROOT / ".venv_clean" / "Scripts" / "python.exe"
    for name, frame, env_paths in cases:
        with tempfile.TemporaryDirectory(prefix="generalized_track2_") as temp:
            d = Path(temp)
            for filename in ["prediction_notebook.ipynb", "model.pkl"]:
                shutil.copy2(PACKAGE / filename, d / filename)
            input_path = d / ("hidden.csv" if env_paths else "test.csv")
            output_path = d / "nested" / "out.csv" if env_paths else d / "predictions.csv"
            if frame is None:
                input_path.write_bytes(b"")
            else:
                frame.to_csv(input_path, index=False)
            env = {k:v for k,v in os.environ.items() if not k.startswith("DATATHON")}
            if env_paths:
                env.update(DATATHON_INPUT_PATH=str(input_path), DATATHON_OUTPUT_PATH=str(output_path))
            result = subprocess.run([str(python), "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                                     "--ExecutePreprocessor.timeout=120", "--output", "executed.ipynb", "prediction_notebook.ipynb"],
                                    cwd=d, env=env, capture_output=True, text=True, timeout=150)
            assert result.returncode == 0, (name, result.stderr[-2500:])
            output = pd.read_csv(output_path)
            assert list(output.columns) == ["prediction"]
            assert len(output) == (0 if frame is None else len(frame))
            numeric = output.prediction.to_numpy(dtype=float)
            assert ((numeric >= 0) & (numeric <= 1)).all() and np.isfinite(numeric).all()
            received = pd.DataFrame() if frame is None else pd.read_csv(input_path)
            difference = 0. if not len(output) else float(np.max(np.abs(output.prediction - expected(received))))
            assert difference < 1e-9, (name,difference)
            checks.append({"case": name, "rows": len(output), "max_difference": difference, "passed": True})
            print(json.dumps(checks[-1]), flush=True)
    assert protocol["C2_hashes"] == {p.name: sha(p) for p in C2.iterdir() if p.is_file()}
    validation = {"passed": True, "isolated_notebook_checks": checks, "singleton_batch_invariance": True,
                  "baseline_C2_max_difference": c2_difference, "C2_files_unchanged": True,
                  "python": str(python), "test_rank_correlation_C2": float(spearmanr(full,c2).statistic),
                  "candidate_flagged_at_0_5": int((full >= .5).sum()), "C2_flagged_at_0_5": int((c2 >= .5).sum())}
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    write_report(validation, b["decision_threshold"], concept)
    print("ALL NOTEBOOK CHECKS PASSED; C2 unchanged", flush=True)
    print(concept.pivot(index="view",columns="model",values="ap").round(4).to_string(), flush=True)


def write_report(validation, threshold, concept):
    summary = pd.read_csv(OUT / "summary.csv")
    country = pd.read_csv(OUT / "country_holdout.csv")
    def md_table(df):
        # Avoid an authoring dependency on tabulate in the organiser environment.
        lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"]*len(df.columns)) + " |"]
        for row in df.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, (float,np.floating)) else str(v) for v in row) + " |")
        return "\n".join(lines)
    views = ["standard","missing_5pct","missing_10pct","bursts_x3","bursts_x0.3","older_accounts_x3","new_devices_x3","non_SG_x3","fraud_prior_0.005","fraud_prior_0.01"]
    table = summary.pivot(index="view",columns="model",values="ap").loc[views,["c2_recipe","unadjusted_transductive","train_only","train_only_no_knn"]].reset_index()
    concept_table = concept.pivot(index="view",columns="model",values="ap")[["c2_recipe","train_only"]].reset_index()
    standard = summary.query('view == "standard" and model in ["c2_recipe","train_only"]')[["model","ap","f1","recall"]]
    readme = f"""# Train-only candidate for Track 2

This is a separate candidate. The scored C2 final submission is unchanged. This candidate has no public submission hash or measured public/private score.

## What changed

- Removed all public-leaderboard-tuned group shifts, high-value look-alike rules and country-burst corrections.
- Retrained the same three model families with category amount medians from training alone. No public-test features are read during fitting.
- Kept the 56% v5 / 24% interaction / 20% GALR logit blend and their existing regularisation settings.
- Kept nearest-neighbour features: removing them slightly reduced local average precision.
- Estimated a decision threshold from training calibration splits; final threshold **{threshold:.6f}**. A fixed logit shift maps it to 0.5. This preserves ranking, but outputs are decision scores rather than calibrated probabilities.
- Added empty-input and nested-output-directory handling.

## What the experiment shows

Five stratified outer folds for each of seeds 43017 and 92041. Each outer validation row is unseen by its fitted model; all preprocessing is fitted within the training fold for the candidate. The original transductive recipe uses outer-training plus unlabeled validation category medians as a control. The actual C2 corrections are copied from the final notebook. Their full-data implementation reproduces the scored C2 CSV within {validation['baseline_C2_max_difference']:.2e}.

The rows and labels have been used in extensive earlier development. Different seeds reuse the same 353 fraud labels. These are local diagnostic estimates, not an untouched test set or statistically independent experiments. All candidate settings were fixed before this run; no hyperparameter sweep or new public submission was performed.

Mean average precision (the repository's PR-AUC metric):

{md_table(table)}

Missingness masks an extra 5% or 10% of each feature in validation only. Other population views reweight existing labeled rows. Population views assume the conditional fraud mechanism remains unchanged; their absolute AP values are not directly comparable across different prevalences. Compare models within each row.

The main improvement comes from removing public corrections, not replacing test-derived medians: the unadjusted control and train-only candidate are nearly identical. A lower train-CV score from C2 is not proof of leaderboard overfitting, because its rules were intended for a different distribution.

Threshold results on outer validation:

{md_table(standard)}

For the candidate, every outer threshold is chosen on a separate 25% split of outer training, using another model fit on the remaining 75%. The final threshold is the median of ten thresholds. Refitting on all training data can change score calibration. There is no proof this threshold is optimal on private data. C2 uses its original fixed shift; control-model F1 results in the CSV use the candidate threshold diagnostically and are not separately tuned.

## If the conditional fraud rate changes

After fixing the candidate, a diagnostic reweights only legitimate examples in specified groups by 3x or 10x. This represents one possible dilution of risk, not private-label knowledge. No model choice or parameter is changed based on this check.

{md_table(concept_table)}

These checks only amplify observed legitimate examples. They do not generate unseen fraud mechanisms or the public set's suspected synthetic look-alikes. Neither model is guaranteed robust to arbitrary concept shift. C2 has higher AP when legitimate bursts are multiplied tenfold (0.1349 vs 0.1251). It also has better F1 when legitimate older-account new-device rows are multiplied 3x or 10x, even though the candidate has higher AP. Public corrections may help if their patterns recur in private data; removing them can sacrifice that advantage. This candidate is less dependent on public feedback, not universally more robust.

Country holdouts exclude the listed countries from all fitting and evaluate on them as unseen categories:

{md_table(country)}

## Files and validation

Upload `prediction_notebook.ipynb` with **this folder's** `model.pkl` if choosing this candidate. Do not mix it with C2's model. No requirements file is needed beyond the preinstalled numpy, pandas, scikit-learn and joblib libraries. `reference_predictions_unscored.csv` is a local reference and should not be uploaded as part of the notebook package.

Six isolated notebook executions passed with the organiser-version Python environment: full input, shuffled input without id, malformed/missing columns, all-missing rows, header-only input and zero-byte input. Output is one `prediction` column, finite values in [0,1], and input row order. Singleton and batch predictions agree. Candidate flags {validation['candidate_flagged_at_0_5']} of 12,000 local test rows at 0.5; C2 flags {validation['C2_flagged_at_0_5']}. The flagged rate is diagnostic, not a target inferred from hidden labels.

No Linux-platform run, new public evaluation or private evaluation was performed. C2 file hashes were checked before and after all work and are unchanged.

Evidence: `Track 2/experiments/generalization/` contains protocol, OOF caches, detailed metrics, thresholds, country holdouts, baseline equivalence and notebook validation. Reproduce from `Track 2` using `.venv_clean/Scripts/python.exe src/generalization_check.py` then `src/generalization_validate.py` with the same interpreter.

Validation follows scikit-learn guidance on [keeping preprocessing within training folds](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) and [separating classifier fitting from threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html).
"""
    (PACKAGE / "README.md").write_text(readme, encoding="utf-8")
    (OUT / "REPORT.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
