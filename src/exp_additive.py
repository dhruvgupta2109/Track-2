import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); from common import *; import models
from features_prod import make_features; from constants import fit_constants
import lightgbm as lgb, xgboost as xgb
from sklearn.metrics import average_precision_score as ap
tr, te = load(); y = tr[LABEL].values; R = 5; splits = cv_splits(y, n_repeats=R)
w = np.load("experiments/adv_weights.npy")
GB_COLS = ["transaction_amount","transaction_hour","transactions_last_24h","spend_last_24h","account_age","new_device","transactions_last_1h",
           "amt_rel_cat","avg_spend","amt_rel_avg","t1_share"] + CATS
MONO = {"transactions_last_1h": 1, "transactions_last_24h": 1, "new_device": 1, "account_age": -1}
def gb_prep(a, b):
    K = fit_constants(a); A = make_features(a, K)[GB_COLS]; B = make_features(b, K)[GB_COLS]
    for c in CATS:
        cats = sorted(A[c].dropna().unique()); A[c] = pd.Categorical(A[c], cats); B[c] = pd.Categorical(B[c], cats)
    return A, B
def lgb_f(**p):
    def fp(a, b):
        A, B = gb_prep(a, b); q = dict(p)
        if q.pop("mono", False): q["monotone_constraints"] = [MONO.get(c, 0) for c in GB_COLS]
        return lgb.LGBMClassifier(**q, verbose=-1, n_jobs=4).fit(A, a[LABEL]).predict_proba(B)[:, 1]
    return fp
def xgb_f(**p):
    def fp(a, b):
        A, B = gb_prep(a, b)
        return xgb.XGBClassifier(**p, enable_categorical=True, tree_method="hist", n_jobs=4).fit(A, a[LABEL]).predict_proba(B)[:, 1]
    return fp
def lr_bin(C=0.1):
    # LR + one-hot of velocity counts / age bins (piecewise-constant shape functions) on top of v2 features
    def add(d):
        d["t1c"] = d.transactions_last_1h.clip(upper=6).map(lambda v: np.nan if pd.isna(v) else str(int(v)))
        d["t24c"] = pd.cut(d.transactions_last_24h, [0,7,9,11,13,15,40]).astype(str).replace("nan", np.nan)
        d["agec"] = pd.cut(d.account_age, [-1,30,90,180,365,730,1500,2500,3999,5000]).astype(str).replace("nan", np.nan)
        return d
    def fp(a, b):
        K = fit_constants(a); A = add(make_features(a, K)); B = add(make_features(b, K))
        cat = models.LR_CAT + ["t1c","t24c","agec"]
        m = models.lr_pipeline(C, models.LR_NUM, cat).fit(A[cat + models.LR_NUM], a[LABEL])
        return m.predict_proba(B[cat + models.LR_NUM])[:, 1]
    return fp
exps = {
 "lgb_stumps": lgb_f(n_estimators=1500, learning_rate=0.01, max_depth=1, num_leaves=2, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, cat_smooth=20),
 "lgb_depth2": lgb_f(n_estimators=1000, learning_rate=0.01, max_depth=2, num_leaves=4, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, cat_smooth=20),
 "lgb_depth2_mono": lgb_f(n_estimators=1000, learning_rate=0.01, max_depth=2, num_leaves=4, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, cat_smooth=20, mono=True),
 "xgb_depth2": xgb_f(n_estimators=800, learning_rate=0.015, max_depth=2, min_child_weight=3, subsample=0.8, colsample_bytree=0.7, reg_lambda=5),
 "lr_bin": lr_bin(0.1),
}
for n, fp in exps.items():
    print(n); oof, s = run_cv(fp, tr, splits, R); np.save(f"experiments/oof_{n}.npy", oof)
    s["pr_auc_testweighted"] = float(np.mean([ap(y, oof[r], sample_weight=w) for r in range(R)])); print("   test-weighted", round(s["pr_auc_testweighted"], 4))
    log_result(n, s, "additive models round")
