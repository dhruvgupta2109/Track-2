"""Shared data loading, feature engineering and CV harness for Track 2 (TrustGuard).

Everything a prediction notebook needs (make_features + constants) must be row-local:
the platform test file has no id column and shuffled rows.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, brier_score_loss

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL = "fraud"
CATS = ["merchant_category", "country", "transaction_channel"]
NUMS = ["transaction_amount", "transaction_hour", "transactions_last_24h", "spend_last_24h",
        "account_age", "new_device", "transactions_last_1h"]
RAW = NUMS[:2] + CATS + NUMS[2:]


def load():
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "test.csv"))
    return tr, te


def cv_splits(y, n_splits=5, n_repeats=5, seed=2026):
    return list(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(np.zeros(len(y)), y))


def inject_missing(df, rate, rng, cols=RAW):
    """Blank out values at a given per-column rate (simulates the test set's higher missingness)."""
    df = df.copy()
    for c in cols:
        m = rng.random(len(df)) < rate
        df.loc[m, c] = np.nan
    return df


def metrics(y, p, w=None):
    out = {"pr_auc": average_precision_score(y, p, sample_weight=w), "roc_auc": roc_auc_score(y, p, sample_weight=w)}
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    out["logloss"] = log_loss(y, pc, sample_weight=w)
    out["brier"] = brier_score_loss(y, pc, sample_weight=w)
    return out


def run_cv(fit_predict, tr, splits, n_repeats, extra_missing=0.0, seed=0, verbose=True):
    """fit_predict(train_df, valid_df) -> valid probabilities. Returns per-repeat OOF matrix and summary.

    extra_missing > 0 additionally blanks values in the *validation* fold only, to emulate test missingness.
    """
    y = tr[LABEL].values
    oof = np.full((n_repeats, len(tr)), np.nan)
    rng = np.random.default_rng(seed)
    k = len(splits) // n_repeats
    for i, (a, b) in enumerate(splits):
        va = tr.iloc[b]
        if extra_missing:
            va = inject_missing(va, extra_missing, rng)
        oof[i // k, b] = fit_predict(tr.iloc[a], va)
    per_rep = [average_precision_score(y, oof[r]) for r in range(n_repeats)]
    fold_ap = [average_precision_score(y[b], oof[i // k, b]) for i, (a, b) in enumerate(splits)]
    summ = {"pr_auc_mean": float(np.mean(per_rep)), "pr_auc_rep_std": float(np.std(per_rep)),
            "fold_ap_std": float(np.std(fold_ap)), "roc_auc": float(roc_auc_score(y, oof.mean(0))),
            "pr_auc_avgoof": float(average_precision_score(y, oof.mean(0)))}
    if verbose:
        print(json.dumps({k_: round(v, 4) for k_, v in summ.items()}))
    return oof, summ


def log_result(name, summ, notes=""):
    path = os.path.join(ROOT, "experiments", "results.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps({"name": name, **{k: round(v, 5) for k, v in summ.items()}, "notes": notes}) + "\n")
