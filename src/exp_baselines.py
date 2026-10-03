import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); from common import *
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
tr, te = load(); y = tr[LABEL].values
R = 5; splits = cv_splits(y, n_repeats=R)

def prep_cat(d):
    d = d[RAW].copy()
    for c in CATS: d[c] = pd.Categorical(d[c], categories=sorted(tr[c].dropna().unique()))
    return d

def lgb_fp(params):
    def fp(a, b):
        m = lgb.LGBMClassifier(**params, verbose=-1, n_jobs=4).fit(prep_cat(a), a[LABEL])
        return m.predict_proba(prep_cat(b))[:, 1]
    return fp

def lr_feats(d):
    d = d[RAW].copy()
    for c in ["transaction_amount", "spend_last_24h", "account_age"]: d[c] = np.log1p(d[c])
    return d
def lr_fp(a, b):
    num = [c for c in RAW if c not in CATS]
    ct = ColumnTransformer([("c", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(handle_unknown="ignore")), CATS),
                            ("n", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num)])
    m = make_pipeline(ct, LogisticRegression(C=1.0, max_iter=2000)).fit(lr_feats(a), a[LABEL])
    return m.predict_proba(lr_feats(b))[:, 1]

def hgb_fp(a, b):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40,
                                       l2_regularization=1.0, categorical_features="from_dtype", random_state=0)
    m.fit(prep_cat(a), a[LABEL]); return m.predict_proba(prep_cat(b))[:, 1]

exps = {
 "lr_raw_log": lr_fp,
 "lgb_default": lgb_fp(dict()),
 "lgb_reg": lgb_fp(dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5)),
 "hgb_reg": hgb_fp,
}
for n, fp in exps.items():
    print(n); oof, s = run_cv(fp, tr, splits, R); log_result(n, s, "raw features baseline 5x5 RSKF")
    np.save(f"experiments/oof_{n}.npy", oof)
