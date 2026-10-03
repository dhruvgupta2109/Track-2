"""Write the inference-only prediction notebook for a package dir. The feature cell is src/features_prod.py verbatim."""
import sys, json
pkg = sys.argv[1]
feat = open("src/features_prod.py").read().strip()
feat = "\n".join(l for l in feat.splitlines() if not l.startswith("import "))
def code(s): return {"cell_type": "code", "metadata": {}, "source": s, "execution_count": None, "outputs": []}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}},
      "cells": [
        md("# TrustGuard - prediction notebook (inference only)\n\nLoads the trained artifact `model.pkl` uploaded next to this notebook, rebuilds the same row-local features used in training "
           "(identical code), and writes one fraud probability in [0, 1] per test row, in the input row order. No training happens here.\n\n"
           "Robustness: missing columns/values, unknown categories, out-of-range or non-numeric values are all handled (clipped to the training range / imputed)."),
        code('import os\nimport joblib\nimport numpy as np\nimport pandas as pd\n\nTEST_PATH = os.environ.get("DATATHON_INPUT_PATH", "test.csv")\nOUTPUT_PATH = os.environ.get("DATATHON_OUTPUT_PATH", "predictions.csv")'),
        code(feat),
        code('bundle = joblib.load("model.pkl")  # uploaded in card 2, next to this notebook\nmodel, features, K = bundle["model"], bundle["features"], bundle["constants"]\n\ntest_df = pd.read_csv(TEST_PATH)\nX_test = make_features(test_df, K)[features]\npreds = model.predict_proba(X_test)[:, 1]  # probability of fraud'),
        code('preds = np.nan_to_num(np.asarray(preds, dtype=float), nan=float(bundle.get("prior", 0.0177)))\npreds = np.clip(preds, 0.0, 1.0)\nassert len(preds) == len(test_df)\n\n'
             '# Platform test file has no id column (template): write `prediction` only. If an id column is present (local test.csv), keep it.\n'
             'out = pd.DataFrame({"prediction": preds})\nif "id" in test_df.columns:\n    out.insert(0, "id", test_df["id"].values)\nout.to_csv(OUTPUT_PATH, index=False)\nprint(f"wrote {len(out)} predictions to {OUTPUT_PATH}; columns={list(out.columns)}")'),
      ]}
json.dump(nb, open(f"{pkg}/prediction_notebook.ipynb", "w"), indent=1)
print("notebook written")
