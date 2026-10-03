import sys, joblib, numpy as np, pandas as pd
sys.path.insert(0, "src"); from common import *; from features import fit_constants
exec(open("packages/v1_lr_eng/_features.py").read())   # use the exact notebook feature code
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
tr, te = load()
NUMF = ["log_amt","log_spend","log_age","new_device","transactions_last_24h","transactions_last_1h",
        "amt_rel_cat","avg_spend","amt_rel_avg","young","t1_hi","t24_hi","burst","t1_share","night","foreign"]
CATF = CATS + ["hc"]
K = fit_constants(tr)
ct = ColumnTransformer([("c", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(handle_unknown="ignore")), CATF),
                        ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMF)])
model = make_pipeline(ct, LogisticRegression(C=0.1, max_iter=5000)).fit(make_features(tr, K)[CATF + NUMF], tr[LABEL])
joblib.dump({"model": model, "features": CATF + NUMF, "constants": K}, "packages/v1_lr_eng/model.pkl")
p = model.predict_proba(make_features(te, K)[CATF + NUMF])[:, 1]
pd.DataFrame({"id": te.id, "prediction": p}).to_csv("submissions/sub_v1_lr_eng.csv", index=False)
print("ok", p.min(), p.max(), p.mean())
