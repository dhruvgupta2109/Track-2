"""Flexible logistic-regression factory for the round-4 experiments."""
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from features_prod import make_features
from constants import fit_constants
import models

BASE_NUM = models.LR_NUM_V3
BASE_CAT = models.LR_CAT


def add_inter(d, fams):
    b = d.burst.fillna(0); nd = d.new_device.fillna(0); yg = d.young.fillna(0); nt = d.night; fr = d.foreign.fillna(0)
    la = d.log_age.fillna(d.log_age.median()); ar = d.amt_rel_cat.fillna(0)
    big = (d.amt_rel_cat > 1.0).astype(float)
    cash = d.merchant_category.isin(["cash_transfer"]).astype(float); bank = (d.transaction_channel == "bank_transfer").astype(float)
    cols = []
    def put(n, v): d[n] = v; cols.append(n)
    if "burst" in fams:
        put("b_x_logage", b * (la - 6.5)); put("b_x_night", b * nt); put("b_x_cash", b * cash); put("b_x_bank", b * bank)
        put("b_x_nd", b * nd); put("b_x_amt", b * ar)
    if "nd" in fams:
        put("nd_x_logage", nd * (la - 6.5)); put("nd_x_amt", nd * ar); put("nd_x_foreign", nd * fr); put("nd_x_night", nd * nt)
    if "big" in fams:
        put("big_x_young", big * yg); put("big_x_nd", big * nd); put("amt_x_logage", ar * (la - 6.5))
    if "foreign" in fams:
        put("fr_x_amt", fr * ar); put("fr_x_young", fr * yg); put("fr_x_night", fr * nt)
    if "sigcount" in fams:
        put("sig_count", nd + b + yg + fr + big)
    return d, cols


def fit_predict(a, b, C=0.1, num=None, cat=None, fams=(), spline=None, bins=None, penalty="l2", l1_ratio=None, freq=False, cross=()):
    num = list(num or BASE_NUM); cat = list(cat or BASE_CAT)
    K = fit_constants(a); A = make_features(a, K); B = make_features(b, K)
    if fams:
        A, ic = add_inter(A, fams); B, _ = add_inter(B, fams); num += ic
    for c1, c2 in cross:
        n = f"{c1}__{c2}"; A[n] = A[c1].astype(str) + "|" + A[c2].astype(str); B[n] = B[c1].astype(str) + "|" + B[c2].astype(str); cat.append(n)
    if freq:
        for c in ["merchant_category", "country", "transaction_channel"]:
            f = A[c].value_counts(normalize=True); A[c + "_freq"] = A[c].map(f).astype(float); B[c + "_freq"] = B[c].map(f).astype(float); num.append(c + "_freq")
    tf = [("c", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(handle_unknown="ignore", min_frequency=10 if cross else None)), cat),
          ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num)]
    if spline:
        cols, k = spline
        tf.append(("s", make_pipeline(SimpleImputer(strategy="median"), SplineTransformer(n_knots=k, degree=3, extrapolation="constant"), StandardScaler()), cols))
    if bins:
        cols, k = bins
        tf.append(("q", make_pipeline(SimpleImputer(strategy="median"), KBinsDiscretizer(n_bins=k, encode="onehot", strategy="quantile")), cols))
    solver = "saga" if penalty in ("l1", "elasticnet") else "lbfgs"
    lr = LogisticRegression(C=C, penalty=penalty, l1_ratio=l1_ratio, solver=solver, max_iter=5000 if solver == "lbfgs" else 3000)
    m = make_pipeline(ColumnTransformer(tf), lr).fit(A, a["fraud"])
    return m.predict_proba(B)[:, 1]


def F(**kw):
    return lambda a, b: fit_predict(a, b, **kw)
