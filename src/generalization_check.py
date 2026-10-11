"""Small train-only candidate and matched robustness checks; never modifies C2."""
from pathlib import Path
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, recall_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from build_v5 import fit_v5, predict_v5
from codex_polylab import fit_poly, poly_frame, fit_galr, galr_frame, blend
from features_prod import sanitize, make_features
from common import RAW, inject_missing

ROOT = Path(__file__).resolve().parents[1]
C2 = ROOT.parent / "FINAL_SUBMISSION_Track2_C2"
OUT = ROOT / "experiments" / "generalization"
PACKAGE = ROOT.parent / "GENERALIZED_Track2_candidate"
SEEDS = [43017, 92041]
NAMES = ["c2_recipe", "unadjusted_transductive", "train_only", "train_only_no_knn"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def notebook_runtime():
    nb = json.loads((C2 / "prediction_notebook.ipynb").read_text(encoding="utf-8"))
    env = {"np": np, "pd": pd}
    for cell in nb["cells"][2:5]:
        exec("".join(cell["source"]), env)
    # Use the actual C2 rules, rather than approximating them for the comparison.
    cell5 = "".join(nb["cells"][5]["source"])
    rules = cell5[cell5.index("SHIFT_BURST_YOUNG ="):]
    rules += "\n" + "".join(nb["cells"][6]["source"])
    rules = "\n".join(line for line in rules.splitlines() if not line.startswith("print("))
    return nb, env, rules


def adjustments(df, bundle, p, runtime, rules):
    env = dict(runtime, X_all=make_features(df, bundle["constants"]), preds=p.copy())
    exec(rules, env)
    return env["preds"]


def fit_members(a, knn=True, context=None):
    v = fit_v5(a, knn=knn)
    pm, pc = fit_poly(sanitize(a, v["constants"]), "original", context=context)
    gm = fit_galr(sanitize(a, v["constants"]))
    return v, pm, pc, gm


def predict_members(members, frame):
    v, pm, pc, gm = members
    clean = sanitize(frame, v["constants"])
    return blend(predict_v5(v, clean), pm.predict_proba(poly_frame(clean, pc))[:, 1],
                 gm.predict_proba(galr_frame(clean))[:, 1])


def threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f)])


def weights(frame, y):
    burst = frame.transactions_last_24h.ge(16).to_numpy()
    old = frame.account_age.gt(365).to_numpy()
    nd = frame.new_device.eq(1).to_numpy()
    foreign = (frame.country.notna() & frame.country.ne("SG")).to_numpy()
    result = {"standard": np.ones(len(frame)),
              "bursts_x3": np.where(burst, 3., 1.),
              "bursts_x0.3": np.where(burst, .3, 1.),
              "older_accounts_x3": np.where(old, 3., 1.),
              "new_devices_x3": np.where(nd, 3., 1.),
              "non_SG_x3": np.where(foreign, 3., 1.)}
    for prior in [.005, .01]:
        current = y.mean()
        result[f"fraud_prior_{prior}"] = np.where(y == 1, prior / current, (1-prior) / (1-current))
    return result


def record(seed, view, name, y, p, w, decision):
    return {"seed": seed, "view": view, "model": name,
            "ap": float(average_precision_score(y, p, sample_weight=w)),
            "brier": float(brier_score_loss(y, p, sample_weight=w)),
            "f1": float(f1_score(y, decision, sample_weight=w, zero_division=0)),
            "recall": float(recall_score(y, decision, sample_weight=w, zero_division=0)),
            "effective_n": float(w.sum()**2 / (w @ w)),
            "weighted_fraud_rate": float(np.average(y, weights=w))}


def main():
    OUT.mkdir(exist_ok=True)
    original_hashes = {p.name: sha(p) for p in C2.iterdir() if p.is_file()}
    protocol = {"seeds": SEEDS, "folds": 5, "models": NAMES,
                "train_sha256": sha(ROOT / "train.csv"), "code_sha256": sha(Path(__file__)),
                "C2_hashes": original_hashes,
                "primary_candidate": "train_only: same blend weights and regularisation; no public rules or test-feature medians",
                "selection": "No parameter search. No-knn is a diagnostic; train_only is the predeclared artifact.",
                "threshold": "For each outer fold, fit a separate train-only model on 75% of outer training; choose F1 threshold on the remaining 25%. Evaluate threshold on outer validation. Final uses median of these 10 thresholds.",
                "limitations": "Repeated splits reuse the same 353 positive labels and do not undo historical selection. Population weights assume unchanged P(y|x); they cannot verify new fraud mechanisms. Missingness is feature masking, not concept shift. No private/public evaluation."}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    nb, runtime, rules = notebook_runtime()
    tr = pd.read_csv(ROOT / "train.csv")
    y = tr.fraud.to_numpy()
    rows, chosen_thresholds = [], []
    started = time.monotonic()
    with threadpool_limits(limits=2):
        for seed in SEEDS:
            scores = {v: {name: np.full(len(tr), np.nan) for name in NAMES}
                      for v in ["standard", "missing_5pct", "missing_10pct"]}
            fold_threshold = np.full(len(tr), np.nan)
            for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y), 1):
                A, B = tr.iloc[a], tr.iloc[b]
                members = fit_members(A)
                vm = fit_v5(A, knn=False)
                no_knn = (vm, *members[1:])
                clean_A = sanitize(A, members[0]["constants"])
                trans_pm, trans_pc = fit_poly(clean_A, "original", context=B[RAW])
                trans = (members[0], trans_pm, trans_pc, members[3])
                # An inner calibration split keeps the evaluated labels out of threshold choice.
                ia, ib = train_test_split(np.arange(len(A)), test_size=.25, stratify=A.fraud,
                                         random_state=seed + fold)
                inner = fit_members(A.iloc[ia])
                t = threshold(A.iloc[ib].fraud.to_numpy(), predict_members(inner, A.iloc[ib]))
                chosen_thresholds.append(t)
                fold_threshold[b] = t
                views = {"standard": B}
                for rate in [.05, .10]:
                    views[f"missing_{int(rate*100)}pct"] = inject_missing(B, rate, np.random.default_rng(seed + fold*100 + int(rate*100)))
                for view, frame in views.items():
                    base = predict_members(trans, frame)
                    scores[view]["unadjusted_transductive"][b] = base
                    scores[view]["c2_recipe"][b] = adjustments(frame, members[0], base, runtime, rules)
                    scores[view]["train_only"][b] = predict_members(members, frame)
                    scores[view]["train_only_no_knn"][b] = predict_members(no_knn, frame)
                print(f"seed={seed} fold={fold}/5 threshold={t:.4f} elapsed={time.monotonic()-started:.1f}s", flush=True)
            np.savez_compressed(OUT / f"oof_{seed}.npz", **{f"{view}__{name}": p for view, preds in scores.items() for name,p in preds.items()}, threshold=fold_threshold)
            for view, preds in scores.items():
                for name, p in preds.items():
                    assert np.isfinite(p).all()
                    decisions = p >= (.5 if name == "c2_recipe" else fold_threshold)
                    # F1 for the transductive and no-knn controls uses the candidate threshold diagnostically.
                    rows.append(record(seed, view, name, y, p, np.ones(len(tr)), decisions))
                    if view == "standard":
                        for weight_view, w in weights(tr, y).items():
                            if weight_view != "standard":
                                rows.append(record(seed, weight_view, name, y, p, w, decisions))
            print(pd.DataFrame(rows).query('seed == @seed and view == "standard"')[["model","ap","f1"]].to_string(index=False), flush=True)
        # Held-country-out evaluation tests transfer to country categories unseen during fitting.
        country_rows = []
        for countries in [("AU", "JP"), ("MY", "ID")]:
            held = tr.country.isin(countries)
            A, B = tr.loc[~held], tr.loc[held]
            m = fit_members(A)
            n = (fit_v5(A, knn=False), *m[1:])
            for name, members in [("train_only", m), ("train_only_no_knn", n)]:
                p = predict_members(members, B)
                country_rows.append({"held_countries": "/".join(countries), "model": name,
                                     "rows": len(B), "positives": int(B.fraud.sum()),
                                     "ap": float(average_precision_score(B.fraud, p))})
        pd.DataFrame(country_rows).to_csv(OUT / "country_holdout.csv", index=False)
        pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
        final = fit_members(tr)
        v, pm, pc, gm = final
        v.update(poly_model=pm, poly_constants=pc, galr_model=gm,
                 galr_columns=list(galr_frame(sanitize(tr, v["constants"])).columns),
                 blend_w_poly=.3, blend_w_galr=.2, decision_threshold=float(np.median(chosen_thresholds)),
                 provenance="train.csv only; no public corrections")
        PACKAGE.mkdir(exist_ok=True)
        joblib.dump(v, PACKAGE / "model.pkl")
        write_notebook(nb, v["decision_threshold"])
        summary = pd.DataFrame(rows).groupby(["view", "model"])[["ap", "brier", "f1", "recall"]].mean().reset_index()
        summary.to_csv(OUT / "summary.csv", index=False)
        (OUT / "thresholds.json").write_text(json.dumps({"inner_thresholds": chosen_thresholds, "final_threshold": v["decision_threshold"]}, indent=2), encoding="utf-8")
        manifest = {"train_sha256": protocol["train_sha256"], "model_sha256": sha(PACKAGE / "model.pkl"),
                    "notebook_sha256": sha(PACKAGE / "prediction_notebook.ipynb"), "protocol": str(OUT / "protocol.json"),
                    "threshold": v["decision_threshold"], "public_score": None, "private_score": None}
        (PACKAGE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    assert original_hashes == {p.name: sha(p) for p in C2.iterdir() if p.is_file()}, "C2 files changed"
    print(summary.query('view in ["standard", "missing_10pct", "bursts_x3"]')[['view','model','ap','f1']].to_string(index=False), flush=True)
    print("Candidate saved:", PACKAGE, "seconds:", round(time.monotonic()-started, 1), flush=True)


def write_notebook(original, decision_threshold):
    import copy
    nb = copy.deepcopy(original)
    nb["cells"][0]["source"] = ("""# TrustGuard: train-only prediction notebook

This version keeps the three-model blend but removes the group adjustments tuned on the public leaderboard. All training constants, including category amount medians, come from `train.csv` only. The saved model is already trained.

The blend is 56% v5 logistic regression, 24% interaction logistic regression and 20% GALR, combined in logit space. The nearest-neighbour feature uses stored training examples. The model settings and blend weights are the same as before.

Keep `model.pkl` next to this notebook. Input defaults to `test.csv`; output defaults to `predictions.csv`. The platform can override these through `DATATHON_INPUT_PATH` and `DATATHON_OUTPUT_PATH`.

The operating threshold was estimated using calibration splits inside cross-validation on training data. A constant logit shift maps that threshold to 0.5 without changing the ranking. The output is a decision score, not a calibrated fraud probability.

This reduces dependence on the public test set. It does not guarantee a higher private score, particularly if the public-test patterns also occur in private data. See the accompanying README for local validation results and limits.

The output has one `prediction` column, with one score per input row in the original order. Missing values, unseen categories and numeric outliers are handled using the saved preprocessing.
""").splitlines(keepends=True)
    nb["cells"][5]["source"] = """bundle = joblib.load("model.pkl")
try:
    test_df = pd.read_csv(TEST_PATH)
except pd.errors.EmptyDataError:
    test_df = pd.DataFrame()

if len(test_df) == 0:
    preds = np.empty(0, dtype=float)
else:
    X_all = make_features(test_df, bundle["constants"])
    X_all = add_v5_features(X_all, bundle["constants"], knn=bundle["v5_knn"])
    p_v5 = bundle["model"].predict_proba(X_all[bundle["features"]])[:, 1]
    p_poly = bundle["poly_model"].predict_proba(poly_features(X_all, bundle["poly_constants"]))[:, 1]
    p_galr = bundle["galr_model"].predict_proba(basis(X_all)[bundle["galr_columns"]])[:, 1]
    to_logit = lambda q: np.log(np.clip(q, 1e-9, 1-1e-9) / (1-np.clip(q, 1e-9, 1-1e-9)))
    z = .56 * to_logit(p_v5) + .24 * to_logit(p_poly) + .20 * to_logit(p_galr)
    # Map the training-validation threshold to 0.5 using one constant shift.
    t = bundle["decision_threshold"]
    z -= np.log(t / (1-t))
    preds = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
""".splitlines(keepends=True)
    nb["cells"][6]["cell_type"] = "markdown"
    nb["cells"][6]["source"] = ["The next cell checks the scores and saves the CSV.\n"]
    nb["cells"][6].pop("execution_count", None)
    nb["cells"][6].pop("outputs", None)
    last = "".join(nb["cells"][7]["source"])
    last = last.replace("out.to_csv(OUTPUT_PATH, index=False)", 'os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)\nout.to_csv(OUTPUT_PATH, index=False)')
    nb["cells"][7]["source"] = last.splitlines(keepends=True)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["execution_count"] = None
            c["outputs"] = []
            compile("".join(c["source"]), "candidate_cell", "exec")
    (PACKAGE / "prediction_notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
