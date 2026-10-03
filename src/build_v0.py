"""Fallback package v0: logistic regression on raw (log) features. Training script (not uploaded)."""
import sys, joblib, numpy as np, pandas as pd
sys.path.insert(0, "src"); from common import *
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def make_features(df):
    df = df.copy()
    for c in ["transaction_amount", "spend_last_24h", "account_age"]:
        df[c] = np.log1p(df[c])
    return df

tr, te = load()
features = RAW
num = [c for c in features if c not in CATS]
ct = ColumnTransformer([("c", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(handle_unknown="ignore")), CATS),
                        ("n", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num)])
model = make_pipeline(ct, LogisticRegression(C=1.0, max_iter=2000)).fit(make_features(tr)[features], tr[LABEL])
joblib.dump((model, features), "packages/v0_lr/model.pkl")
p = model.predict_proba(make_features(te)[features])[:, 1]
pd.DataFrame({"id": te.id, "prediction": p}).to_csv("submissions/sub_v0_lr.csv", index=False)
print("ok", p.min(), p.max(), p.mean())
