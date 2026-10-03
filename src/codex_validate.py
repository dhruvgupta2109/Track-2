"""Fail-closed equivalence and isolated-notebook checks for the reviewed R2op."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from r2_runtime import predict_r2
from codex_delivery import REFERENCE_CSV, PACKAGE, ROOT, sha


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def validate(clean_python):
    start = time.monotonic()
    te = pd.read_csv(ROOT / "test.csv")
    ref = pd.read_csv(REFERENCE_CSV).set_index("id").loc[te.id, "prediction"].to_numpy()
    bundle = joblib.load(PACKAGE / "model.pkl")
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    check(sha(PACKAGE / "model.pkl") == manifest["original_model_sha256"], "Model bytes changed")
    checks = []
    with threadpool_limits(limits=2):
        p = predict_r2(te, bundle)
        check(np.max(np.abs(p - ref)) < 1e-9, "Whole-input reference equivalence failed")
        ix = np.random.default_rng(19).choice(len(te), 24, replace=False)
        one = np.concatenate([predict_r2(te.iloc[[i]].drop(columns="id"), bundle) for i in ix])
        check(np.max(np.abs(one - p[ix])) < 1e-9, "Singleton/batch independence failed")
        dup = te.iloc[np.r_[ix, ix]].drop(columns="id").copy()
        dp = predict_r2(dup, bundle)
        check(np.max(np.abs(dp - np.tile(p[ix], 2))) < 1e-9, "Duplicate-input invariance failed")
        reorder = te.iloc[ix, ::-1].assign(unused="ignored")
        check(np.max(np.abs(predict_r2(reorder, bundle) - p[ix])) < 1e-9, "Column invariance failed")
        strings = te.iloc[ix].copy()
        for c in ("merchant_category", "country", "transaction_channel"):
            strings[c] = strings[c].map(lambda v: " " + v.swapcase() + " " if isinstance(v, str) else v)
        check(np.max(np.abs(predict_r2(strings, bundle) - p[ix])) < 1e-9, "Category normalisation failed")
        checks.extend(["unchanged model bytes", "all 12,000 reference scores", "singleton independence",
                       "duplicate independence", "column order/extra column", "category case/whitespace"])

    rng = np.random.default_rng(7)
    shuffled = te.iloc[rng.permutation(len(te))].drop(columns="id")
    edge = pd.DataFrame({"transaction_amount": [0, -5, np.inf, np.nan, 50, "abc"],
                         "transaction_hour": [24, -1, np.nan, 3, 12, 7.6],
                         "merchant_category": ["crypto", None, "luxury", "grocery", " Grocery ", ""],
                         "country": ["XX", "SG", None, "SG", "sg", "SG"],
                         "transaction_channel": ["atm", "ecommerce", "mobile_app", None, "ecommerce", "card_present"],
                         "transactions_last_24h": [0, 0, 1000, np.nan, 3, 2],
                         "spend_last_24h": [0, 10, 1e8, np.nan, -1, 5],
                         "account_age": [0, -3, 1e5, np.nan, 10, 100],
                         "new_device": [2, np.nan, 1, 0, 1, "yes"], "extra_col": 1})
    cases = [("template_default", te, False), ("shuffled_no_id", shuffled, True),
             ("malformed_missing_column", edge, True), ("singleton", te.iloc[[ix[0]]].drop(columns="id"), True),
             ("all_missing", pd.DataFrame({"transaction_amount": [np.nan] * 3}), True),
             ("empty_header", te.iloc[:0].drop(columns="id"), True),
             ("empty_with_id", te.iloc[:0], True), ("empty_zero_bytes", None, True)]

    def execute_case(case):
        name, frame, env_mode = case
        with tempfile.TemporaryDirectory(prefix="codex_r2_verify_") as temp:
            d = Path(temp)
            for f in ("prediction_notebook.ipynb", "model.pkl", "requirements.txt"):
                shutil.copy2(PACKAGE / f, d / f)
            input_path = d / ("hidden.csv" if env_mode else "test.csv")
            if frame is None:
                input_path.write_bytes(b"")
            else:
                frame.to_csv(input_path, index=False)
            output_path = d / "nested" / "new_directory" / "out.csv" if env_mode else d / "predictions.csv"
            env = {k: v for k, v in os.environ.items() if not k.startswith("DATATHON")}
            env["MPLBACKEND"] = "Agg"
            if env_mode:
                env.update(DATATHON_INPUT_PATH=str(input_path), DATATHON_OUTPUT_PATH=str(output_path))
            result = subprocess.run([str(clean_python), "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                                     "--ExecutePreprocessor.timeout=120", "--output", "executed.ipynb", "prediction_notebook.ipynb"],
                                    cwd=d, env=env, capture_output=True, text=True, timeout=180)
            check(result.returncode == 0, f"{name}: notebook failed\n{result.stderr[-2500:]}")
            out = pd.read_csv(output_path)
            expected_cols = ["id", "prediction"] if frame is not None and "id" in frame else ["prediction"]
            check(list(out.columns) == expected_cols, f"{name}: wrong output columns")
            count = 0 if frame is None else len(frame)
            check(len(out) == count, f"{name}: wrong row count")
            numeric_output = out.prediction.to_numpy(dtype=float)
            check(np.isfinite(numeric_output).all() and ((numeric_output >= 0) & (numeric_output <= 1)).all(), f"{name}: invalid predictions")
            if count:
                # CSV serialisation/coercion is part of the actual input contract.
                received = pd.read_csv(input_path)
                expected = predict_r2(received, bundle)
                difference = float(np.max(np.abs(out.prediction.to_numpy() - expected)))
                check(difference < 1e-9, f"{name}: row alignment/chunk equivalence failed")
                if "id" in received:
                    check(out.id.equals(received.id), f"{name}: changed IDs")
            else:
                difference = 0.0
            record = dict(case=name, rows=count, max_difference=difference, passed=True,
                          training_file_present=(d / "train.csv").exists())
            print(json.dumps(record), flush=True)
            return record

    with ThreadPoolExecutor(max_workers=2) as pool:
        cases_passed = list(pool.map(execute_case, cases))
    summary = {"passed": True, "python": str(clean_python), "in_process_checks": checks,
               "isolated_notebook_cases": cases_passed, "seconds": time.monotonic() - start,
               "model_sha256": sha(PACKAGE / "model.pkl"), "notebook_sha256": sha(PACKAGE / "prediction_notebook.ipynb")}
    (ROOT / "experiments" / "codex" / "package_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"ALL CHECKS PASSED ({summary['seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=ROOT / ".venv_clean" / "Scripts" / "python.exe")
    args = parser.parse_args()
    validate(args.python.resolve())
