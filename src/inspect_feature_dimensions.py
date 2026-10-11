"""Inspect the actual notebook inputs, saved transforms and fresh LR.fit calls."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from build_v5 import fit_v5
from codex_polylab import fit_poly, fit_galr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "feature_dimensions"


def inspect_package(folder, data):
    model_path = folder / "model.pkl"
    bundle = joblib.load(model_path)
    nb = json.loads((folder / "prediction_notebook.ipynb").read_text(encoding="utf-8"))
    env = {"np": np, "pd": pd}
    # Definitions only: do not execute file loading, inference or output-writing cells.
    for cell in nb["cells"][2:5]:
        exec("".join(cell["source"]), env)
    X = env["make_features"](data, bundle["constants"])
    intermediate_shape = list(X.shape)
    X = env["add_v5_features"](X, bundle["constants"], knn=bundle["v5_knn"])
    inputs = {"model": X[bundle["features"]],
              "poly_model": env["poly_features"](X, bundle["poly_constants"]),
              "galr_model": env["basis"](X)[bundle["galr_columns"]]}
    result = {"package": str(folder), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "raw_shape": list(data.shape), "raw_columns": list(data.columns),
              "make_features_shape": intermediate_shape, "add_v5_features_shape": list(X.shape), "models": {}}
    for key, frame in inputs.items():
        pipe = bundle[key]
        ct = pipe[0]
        matrix = pipe[:-1].transform(frame)
        estimator = pipe[-1]
        prediction = pipe.predict_proba(frame)
        direct = estimator.predict_proba(matrix)
        assert np.allclose(prediction, direct, rtol=0, atol=1e-12)
        names = pipe[:-1].get_feature_names_out().tolist()
        assert matrix.shape[1] == len(names) == estimator.coef_.shape[1] == estimator.n_features_in_
        branches = []
        for name, transform, columns in ct.transformers_:
            if name == "remainder":
                branches.append({"name": name, "action": transform, "columns": columns})
                continue
            branch = {"name": name, "input_columns": list(columns),
                      "input_count": len(columns), "stages": []}
            partial = frame[columns]
            for step_name, step in transform.steps:
                before = list(partial.shape)
                partial = step.transform(partial)
                item = {"step": step_name, "type": type(step).__name__,
                        "input_shape": before, "output_shape": list(partial.shape)}
                if hasattr(step, "categories_"):
                    item.update(categories=[values.tolist() for values in step.categories_],
                                category_counts=[len(values) for values in step.categories_],
                                drop=step.drop, drop_idx=step.drop_idx_, handle_unknown=step.handle_unknown)
                if hasattr(step, "powers_"):
                    powers = step.powers_
                    item.update(degree=step.degree, include_bias=step.include_bias,
                                interaction_only=step.interaction_only, powers=powers.tolist(),
                                linear=int((powers.sum(axis=1) == 1).sum()),
                                squares=int((powers.max(axis=1) == 2).sum()),
                                cross_products=int(((powers.sum(axis=1) == 2) & (powers.max(axis=1) == 1)).sum()))
                branch["stages"].append(item)
            branches.append(branch)
        result["models"][key] = {"pipeline_input_shape": list(frame.shape),
                                 "pipeline_input_columns": list(frame.columns),
                                 "LR_input_shape": list(matrix.shape), "matrix_type": type(matrix).__name__,
                                 "ordered_LR_feature_names": names, "coef_shape": list(estimator.coef_.shape),
                                 "intercept_shape": list(estimator.intercept_.shape),
                                 "remainder": ct.remainder, "branches": branches}
    return result


def main():
    OUT.mkdir(exist_ok=True)
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    records = []
    with threadpool_limits(limits=2):
        for name in ["FINAL_SUBMISSION_Track2_C2", "GENERALIZED_Track2_candidate"]:
            folder = ROOT.parent / name
            records.append(inspect_package(folder, test))
        # Capture the actual matrix reaching LR.fit in the reproducing training factories.
        calls = []
        original_fit = LogisticRegression.fit
        def traced_fit(self, X, y, *args, **kwargs):
            calls.append({"C": self.C, "X_shape": list(X.shape), "y_shape": list(np.asarray(y).shape),
                          "matrix_type": type(X).__name__})
            return original_fit(self, X, y, *args, **kwargs)
        LogisticRegression.fit = traced_fit
        try:
            main = fit_v5(train)
            interaction, pc = fit_poly(train, "original", context=test)
            threshold = fit_galr(train)
        finally:
            LogisticRegression.fit = original_fit
        # Confirm reconstructed pipelines match the shipped coefficients and dimensions.
        saved = joblib.load(ROOT.parent / "FINAL_SUBMISSION_Track2_C2" / "model.pkl")
        reconstructed = [main["model"], interaction, threshold]
        for call, key, fitted in zip(calls, ["model", "poly_model", "galr_model"], reconstructed):
            call["model"] = key
            call["max_abs_coefficient_difference_vs_C2"] = float(np.max(np.abs(fitted[-1].coef_ - saved[key][-1].coef_)))
            assert call["X_shape"][1] == saved[key][-1].n_features_in_
            assert fitted[:-1].get_feature_names_out().tolist() == saved[key][:-1].get_feature_names_out().tolist()
        assert len(calls) == 3
    evidence = {"training_dataset_shape": list(train.shape), "training_columns": list(train.columns),
                "test_dataset_shape": list(test.shape), "test_columns": list(test.columns),
                "actual_fresh_LR_fit_calls": calls, "packages": records}
    (OUT / "inspection.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    for rec in records:
        print(Path(rec["package"]).name, "feature engineering shapes:", rec["make_features_shape"], rec["add_v5_features_shape"])
        for key, model in rec["models"].items():
            print(key, "pipeline input:", model["pipeline_input_shape"], "LR input:", model["LR_input_shape"], "coef:", model["coef_shape"])
    print("Actual fresh LR.fit calls:", json.dumps(calls))
    print("All ordered names and transform stages:", OUT / "inspection.json")


if __name__ == "__main__":
    main()
