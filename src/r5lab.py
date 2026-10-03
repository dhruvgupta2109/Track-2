"""v5 experiments: v4 feature space + optional targeted extras; also C, weights."""
import numpy as np, pandas as pd
import models
from features_prod import make_features
from constants import fit_constants

def extras(d):
    b = (d.burst == 1).astype(float); yg = (d.young == 1).astype(float); nt = d.night; nd = d.new_device.fillna(0)
    fr = (d.foreign == 1).astype(float); age = d.account_age; mature = (age > 365).astype(float)
    big = (d.amt_rel_cat > 1.0).astype(float); t1 = d.transactions_last_1h
    X = {}
    X["burst_young"] = b * yg
    X["burst_night"] = b * nt
    X["burst_mature"] = b * mature
    X["burst_x_logage"] = b * (d.log_age.fillna(6.67) - 6.5)
    X["big_mature"] = big * mature
    X["big_young"] = big * yg
    X["amt_pos_x_mature"] = d.amt_rel_cat.clip(lower=0).fillna(0) * mature
    X["young_night"] = yg * nt
    X["foreign_young"] = fr * yg
    X["foreign_nd"] = fr * nd
    X["t1_ge3"] = (t1 >= 3).astype(float).where(t1.notna())
    X["t1_hi"] = d.t1_hi
    X["spike_avg"] = d.amt_rel_avg.clip(lower=0)
    X["cash_mature"] = (d.merchant_category == "cash_transfer").astype(float) * mature
    X["cash_burst"] = (d.merchant_category == "cash_transfer").astype(float) * b
    X["domestic_burst"] = (d.country == "SG").astype(float) * b
    X["day_burst_mature"] = b * (1 - nt) * mature
    for k, v in X.items(): d[k] = v
    return d

def fit_predict(a, b, add=(), C=0.1, sample_weight=None):
    num = models.LR_NUM_V4 + list(add); cat = models.LR_CAT
    K = fit_constants(a); A = extras(make_features(a, K)); B = extras(make_features(b, K))
    m = models.lr_pipeline(C, num, cat)
    m.fit(A[cat + num], a["fraud"], logisticregression__sample_weight=sample_weight)
    return m.predict_proba(B[cat + num])[:, 1]

def F(add=(), C=0.1):
    return lambda a, b: fit_predict(a, b, add, C)
