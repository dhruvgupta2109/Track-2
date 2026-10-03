"""Model factories shared by CV experiments and package builds. All take raw dataframes."""
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from features_prod import make_features, CAT_COLS
from constants import fit_constants

LR_NUM = ["log_amt", "log_spend", "log_age", "new_device", "transactions_last_24h", "transactions_last_1h",
          "amt_rel_cat", "avg_spend", "amt_rel_avg", "young", "t1_hi", "t24_hi", "burst", "t1_share", "night", "foreign"]
LR_CAT = CAT_COLS + ["hc"]
# v3: linear 1h-velocity terms dropped (redundant with the 24h burst cluster, hurt CV on 2 seeds); keep a non-burst rapid-fire flag
LR_NUM_V3 = [f for f in LR_NUM if f not in ("transactions_last_1h", "t1_hi", "t1_share")] + ["t1_ge4_nb"]


def lr_pipeline(C=0.1, num=LR_NUM, cat=LR_CAT):
    ct = ColumnTransformer([("c", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(handle_unknown="ignore")), cat),
                            ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num)])
    return make_pipeline(ct, LogisticRegression(C=C, max_iter=5000))


def fit_lr(train, C=0.1, num=LR_NUM, cat=LR_CAT):
    K = fit_constants(train)
    m = lr_pipeline(C, num, cat).fit(make_features(train, K)[cat + num], train["fraud"])
    return {"model": m, "features": cat + num, "constants": K, "prior": float(train["fraud"].mean())}


def predict_bundle(b, df):
    return b["model"].predict_proba(make_features(df, b["constants"])[b["features"]])[:, 1]

# v4: + new_device x log(account_age) and new_device x night
LR_NUM_V4 = LR_NUM_V3 + ["nd_x_logage", "nd_x_night"]
