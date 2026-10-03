import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from evaluate import evaluate
from features_prod import make_features; from constants import fit_constants
from common import CATS
from catboost import CatBoostClassifier
import lightgbm as lgb
COLS = ["transaction_amount","transaction_hour","transactions_last_24h","spend_last_24h","account_age","new_device","transactions_last_1h","amt_rel_cat","amt_rel_avg"] + CATS
def cb(a, b):
    K = fit_constants(a); A = make_features(a, K)[COLS]; B = make_features(b, K)[COLS]
    for c in CATS: A[c] = A[c].fillna("NA"); B[c] = B[c].fillna("NA")
    return CatBoostClassifier(iterations=800, learning_rate=0.03, depth=4, l2_leaf_reg=10, cat_features=CATS, verbose=0, thread_count=6, random_seed=0).fit(A, a.fraud).predict_proba(B)[:, 1]
def lg(a, b):
    K = fit_constants(a); A = make_features(a, K)[COLS]; B = make_features(b, K)[COLS]
    for c in CATS:
        cats = sorted(A[c].dropna().unique()); A[c] = pd.Categorical(A[c], cats); B[c] = pd.Categorical(B[c], cats)
    mono = {"transactions_last_1h": 1, "transactions_last_24h": 1, "new_device": 1, "account_age": -1}
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=5, monotone_constraints=[mono.get(c, 0) for c in COLS], verbose=-1, n_jobs=6).fit(A, a.fraud).predict_proba(B)[:, 1]
if __name__ == "__main__":
    evaluate("catboost_d4", cb)
    evaluate("lgb_mono", lg)
