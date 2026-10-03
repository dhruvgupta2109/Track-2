"""Build a separate, prediction-equivalent R2op delivery; never mutate the winner.

Run from anywhere with the organiser-version Python. The original notebook is
the source of truth for numerical inference. Its row-local computation is lifted
into a function, embedded in the new notebook, and tested against the scored CSV.
"""
from pathlib import Path
import hashlib
import json
import shutil
import textwrap

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent / "Track2_R2op_package"
PACKAGE = ROOT / "packages" / "codex_r2op_reviewed"
REFERENCE_NOTEBOOK = ROOT / "references" / "r2op_original_prediction_notebook.ipynb"
REFERENCE_CSV = ROOT / "submissions" / "tmpl" / "R2op_package_output.csv"
ORIGINAL_MODEL_SHA256 = "60d9e77b6234ee01af848c2dfc95b5fb37b15d38fc845010422dc96b3f36add7"
ORIGINAL_NOTEBOOK_SHA256 = "ce1d3ceeb8d26f70d6bc5ce5932413a74f56ffdcaf6c5fa308ca13ca18115c10"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell(kind, source, index):
    c = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True), "id": f"r2-reviewed-{index:02}"}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def runtime_source(nb):
    definitions = "\n\n".join("".join(nb["cells"][i]["source"]) for i in (2, 3, 4))
    score = "".join(nb["cells"][5]["source"])
    score = score[score.index("X_all = make_features"):]
    score = score.replace("# Prior corrections for the evaluation data (fixed logit shifts, see markdown above)",
                          "base_probability = preds.copy()\n# Fixed group shifts selected using public-board feedback.")
    score += "\ngroup_probability = preds.copy()\n"
    t2 = "".join(nb["cells"][6]["source"])
    # Keep every numerical expression and constant; replace unsupported certainty
    # in explanatory comments with the actual evidence boundary.
    t2 = t2[t2.index("T2_LEVEL, T2_ALPHA"):]
    t2 = t2[:t2.index("# Operating point.")] + "\nranking_probability = 1.0 / (1.0 + np.exp(-logit))\nOPERATING_POINT_SHIFT = 2.188389\npreds = 1.0 / (1.0 + np.exp(-(logit + OPERATING_POINT_SHIFT)))\n"
    computation = "\n".join(s for s in (score + "\n" + t2).splitlines() if not s.startswith("print("))
    header = '''def predict_r2(test_df, bundle, return_details=False):
    """Return R2op scores in input order, with constants fixed across all batches.

    The default is the historically scored operating-point-adjusted output.
    It is a ranking/decision score in [0,1], not a calibrated fraud probability.
    Diagnostics expose the pre-operating-point scores without changing output.
    """
    if len(test_df) == 0:
        empty = np.empty(0, dtype=float)
        return (empty, {}) if return_details else empty
    model, features, K = bundle["model"], bundle["features"], bundle["constants"]
'''
    footer = '''
    preds = np.nan_to_num(np.asarray(preds, dtype=float), nan=float(bundle.get("prior", 0.0177)))
    preds = np.clip(preds, 0.0, 1.0)
    assert len(preds) == len(test_df) and np.isfinite(preds).all()
    if return_details:
        return preds, {"v5": p_v5, "poly": p_poly, "galr": p_galr,
                       "base": base_probability, "group": group_probability,
                       "ranking": ranking_probability, "T2": is_t2,
                       "burst": is_burst, "nd_old": is_ndold}
    return preds
'''
    return '"""Numerically preserved R2op inference, extracted by codex_delivery.py."""\nimport numpy as np\nimport pandas as pd\n\n' + definitions + "\n\n" + header + textwrap.indent(computation, "    ") + "\n" + footer


def build():
    PACKAGE.mkdir(parents=True, exist_ok=True)
    original_notebook = ORIGINAL / "prediction_notebook.ipynb"
    if original_notebook.exists():
        REFERENCE_NOTEBOOK.parent.mkdir(exist_ok=True)
        shutil.copy2(original_notebook, REFERENCE_NOTEBOOK)
    else:
        original_notebook = REFERENCE_NOTEBOOK
    if sha(original_notebook) != ORIGINAL_NOTEBOOK_SHA256:
        raise RuntimeError("Original notebook differs from the audited R2op snapshot")
    nb = json.loads(original_notebook.read_text(encoding="utf-8"))
    runtime = runtime_source(nb)
    compile(runtime, "r2_runtime.py", "exec")
    (ROOT / "src" / "r2_runtime.py").write_text(runtime, encoding="utf-8")
    for name in ("model.pkl", "requirements.txt"):
        destination = PACKAGE / name
        source = ORIGINAL / name if (ORIGINAL / name).exists() else destination
        if name == "model.pkl" and sha(source) != ORIGINAL_MODEL_SHA256:
            raise RuntimeError("Model differs from the audited R2op winner")
        if destination.exists() and sha(destination) != sha(source):
            raise RuntimeError(f"Refusing to overwrite a different existing {destination}")
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    intro = '''# TrustGuard | R2op prediction notebook

**Cookie Squad - Track 2, NTU Datathon 2026.** This notebook loads the trained bundle and predicts; it does not need training data, internet, repository files, or a GPU.

Recorded public PR-AUC: **0.23434** (historical result in the team's log, not a new evaluation). The three LR members use logit weights **0.56 v5 / 0.24 interaction LR / 0.20 GALR**. Fixed group adjustments and the T2 shrinkage rule follow. Parameters were selected using CV and public-board feedback; transfer to the private distribution is uncertain.

The T2 rate of 3.5% is a working hypothesis inferred indirectly from public probes. Test labels and the data generator are unavailable. Synthetic-looking feature blocks are strong distributional evidence; they do not prove random labels or exact fraud rates.

**Score interpretation:** the final fixed +2.188389 logit shift retains R2op's existing output and all rankings. It puts 300/12,000 public-test rows above 0.5. These shifted values are decision scores, not calibrated fraud probabilities. The handbook names F1/recall, but neither supplied document specifies the private evaluator's threshold; no private F1/recall improvement is established. The `ranking` diagnostic exposes scores before that final shift.

Input: `DATATHON_INPUT_PATH` (default `test.csv`). Output: `DATATHON_OUTPUT_PATH` (default `predictions.csv`). ID-less input produces only `prediction`; local input with `id` preserves it. All decisions use fixed training/reference constants and are invariant to row order and batch composition.
'''
    driver = '''from pathlib import Path
import os
import joblib

TEST_PATH = os.environ.get("DATATHON_INPUT_PATH", "test.csv")
OUTPUT_PATH = os.environ.get("DATATHON_OUTPUT_PATH", "predictions.csv")
bundle = joblib.load("model.pkl")
for name in ("model", "poly_model", "galr_model"):
    assert list(bundle[name].classes_) == [0, 1], f"Unexpected class order in {name}"

try:
    test_df = pd.read_csv(TEST_PATH)
except pd.errors.EmptyDataError:
    test_df = pd.DataFrame()

# Bound intermediate neighbour/feature memory for larger private inputs.
CHUNK_SIZE = 2048
parts = [predict_r2(test_df.iloc[start:start + CHUNK_SIZE], bundle)
         for start in range(0, len(test_df), CHUNK_SIZE)]
preds = np.concatenate(parts) if parts else np.empty(0, dtype=float)
assert len(preds) == len(test_df) and np.isfinite(preds).all()
out = pd.DataFrame({"prediction": preds})
if "id" in test_df.columns:
    out.insert(0, "id", test_df["id"].to_numpy())
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT_PATH, index=False)
print(f"wrote {len(out)} predictions; columns={list(out.columns)}; flagged >=0.5: {int((preds >= 0.5).sum())}")
'''
    new = {"nbformat": 4, "nbformat_minor": 5,
           "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                        "language_info": {"name": "python"}},
           "cells": [cell("markdown", intro, 0), cell("code", runtime, 1),
                     cell("markdown", "## Predict in input order\n\nNo batch-dependent quantiles, retraining, or ID lookup. Empty inputs produce a valid empty output.\n", 2),
                     cell("code", driver, 3)]}
    (PACKAGE / "prediction_notebook.ipynb").write_text(json.dumps(new, indent=1), encoding="utf-8")
    import joblib
    import pandas as pd
    import numpy as np
    namespace = {}
    exec(compile(runtime, "r2_runtime.py", "exec"), namespace)
    te = pd.read_csv(ROOT / "test.csv")
    b = joblib.load(PACKAGE / "model.pkl")
    p = namespace["predict_r2"](te, b)
    expected = pd.read_csv(REFERENCE_CSV).set_index("id").loc[te.id, "prediction"].to_numpy()
    difference = float(np.max(np.abs(p - expected)))
    if difference >= 1e-9:
        raise RuntimeError(f"Prediction equivalence failed: {difference}")
    pd.DataFrame({"id": te.id, "prediction": p}).to_csv(PACKAGE / "predictions_R2op.csv", index=False)
    manifest = {"original_public_score": 0.23434, "new_public_evaluation": False,
                "original_model_sha256": ORIGINAL_MODEL_SHA256, "model_sha256": sha(PACKAGE / "model.pkl"),
                "original_notebook_sha256": sha(original_notebook),
                "notebook_sha256": sha(PACKAGE / "prediction_notebook.ipynb"),
                "runtime_sha256": sha(ROOT / "src" / "r2_runtime.py"),
                "train_sha256": sha(ROOT / "train.csv"), "test_sha256": sha(ROOT / "test.csv"),
                "max_absolute_prediction_difference": difference,
                "public_test_flagged_at_half": int((p >= 0.5).sum()),
                "score_semantics": "Operating-point-adjusted decision scores; not calibrated probabilities."}
    (PACKAGE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    build()
