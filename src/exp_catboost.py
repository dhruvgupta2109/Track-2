import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); from common import *
from features_prod import make_features; from constants import fit_constants
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap
tr, te = load(); y = tr[LABEL].values; R = 3; splits = cv_splits(y, n_repeats=R); w = np.load("experiments/adv_weights.npy")
COLS = ["transaction_amount","transaction_hour","transactions_last_24h","spend_last_24h","account_age","new_device","transactions_last_1h",
        "amt_rel_cat","amt_rel_avg"] + CATS
def cb_f(**p):
    def fp(a, b):
        K = fit_constants(a); A = make_features(a, K)[COLS]; B = make_features(b, K)[COLS]
        for c in CATS: A[c] = A[c].fillna("NA"); B[c] = B[c].fillna("NA")
        m = CatBoostClassifier(**p, cat_features=CATS, verbose=0, thread_count=4, random_seed=0).fit(A, a[LABEL])
        return m.predict_proba(B)[:, 1]
    return fp
for n, fp in {"cb_d4": cb_f(iterations=800, learning_rate=0.03, depth=4, l2_leaf_reg=10),
              "cb_d6": cb_f(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=10)}.items():
    oof, s = run_cv(fp, tr, splits, R, verbose=False); s["pr_auc_testweighted"] = np.mean([ap(y, oof[r], sample_weight=w) for r in range(R)])
    print(n, round(s["pr_auc_mean"], 4), "tw", round(s["pr_auc_testweighted"], 4), flush=True); np.save(f"experiments/oof3_{n}.npy", oof); log_result(n, s, "catboost 3x5")
