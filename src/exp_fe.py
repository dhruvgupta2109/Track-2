import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); from common import *; from features import *
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
tr, te = load(); y = tr[LABEL].values
R = 5; splits = cv_splits(y, n_repeats=R)
ENG = ["log_amt","amt_rel_cat","log_spend","avg_spend","amt_rel_avg","log_age","young","t1_hi","t24_hi","burst","t1_share","night","foreign"]

def lr_fp_factory(nums, C=0.3, spline=False):
    def fp(a, b):
        K = fit_constants(a); A = make_features(a, K); B = make_features(b, K)
        nt = [SimpleImputer(strategy="median", add_indicator=True)] + ([SplineTransformer(n_knots=5, degree=3)] if spline else []) + [StandardScaler()]
        ct = ColumnTransformer([("c", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(handle_unknown="ignore")), CATS + ["transaction_hour_c"]),
                                ("n", make_pipeline(*nt), nums)])
        A["transaction_hour_c"] = A.transaction_hour.astype(str); B["transaction_hour_c"] = B.transaction_hour.astype(str)
        m = make_pipeline(ct, LogisticRegression(C=C, max_iter=5000)).fit(A, a[LABEL])
        return m.predict_proba(B)[:, 1]
    return fp

def lgb_fp_factory(cols, params, mono=None):
    def fp(a, b):
        K = fit_constants(a); A = make_features(a, K)[cols]; B = make_features(b, K)[cols]
        for c in CATS:
            if c in cols:
                cats = sorted(a[c].dropna().unique()); A[c] = pd.Categorical(A[c], cats); B[c] = pd.Categorical(B[c], cats)
        p = dict(params)
        if mono: p["monotone_constraints"] = [mono.get(c, 0) for c in cols]; p["monotone_constraints_method"] = "advanced"
        m = lgb.LGBMClassifier(**p, verbose=-1, n_jobs=4).fit(A, a[LABEL])
        return m.predict_proba(B)[:, 1]
    return fp

base_nums = ["log_amt","log_spend","log_age","new_device","transactions_last_24h","transactions_last_1h"]
P = dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5)
LGBCOLS = RAW + ["amt_rel_cat","avg_spend","amt_rel_avg","t1_share"]
MONO = {"transactions_last_1h": 1, "transactions_last_24h": 1, "new_device": 1, "account_age": -1}
exps = {
 "lr_base_hourcat": lr_fp_factory(base_nums),
 "lr_eng": lr_fp_factory(base_nums + ENG),
 "lr_eng_spline": lr_fp_factory(base_nums + ENG, spline=True, C=0.1),
 "lgb_eng": lgb_fp_factory(LGBCOLS, P),
 "lgb_eng_mono": lgb_fp_factory(LGBCOLS, P, MONO),
 "lgb_raw_mono": lgb_fp_factory(RAW, P, MONO),
}
for n, fp in exps.items():
    print(n); oof, s = run_cv(fp, tr, splits, R); np.save(f"experiments/oof_{n}.npy", oof)
    _, s2 = run_cv(fp, tr, splits, R, extra_missing=0.017, verbose=False); s["pr_auc_testmiss"] = s2["pr_auc_mean"]
    print("   with test-like missingness:", round(s2["pr_auc_mean"], 4))
    log_result(n, s, "FE round 1")
