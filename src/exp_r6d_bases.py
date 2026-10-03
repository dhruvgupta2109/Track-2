"""Strict OOF base-score generators on the evaluate.SPLITS (5x5) for stacking. Saves experiments/oof5_<name>.npy."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from evaluate import evaluate
import r6lab, models
from features_prod import make_features
from constants import fit_constants
from common import CATS

TREE_NUM = ["log_amt", "amt_rel_cat", "log_spend", "avg_spend", "amt_rel_avg", "log_age", "new_device", "transactions_last_24h",
            "transactions_last_1h", "transaction_hour", "night", "foreign", "burst", "young"]


def tree_X(a, b):
    K = fit_constants(a); A = make_features(a, K); B = make_features(b, K)
    XA = A[TREE_NUM].copy(); XB = B[TREE_NUM].copy()
    for c in CATS:
        for v in sorted(A[c].dropna().unique()):
            XA[f"{c}={v}"] = (A[c] == v).astype(float); XB[f"{c}={v}"] = (B[c] == v).astype(float)
    med = XA.median()
    return XA.fillna(med), XB.fillna(med)


def et(a, b, **kw):
    XA, XB = tree_X(a, b)
    p = dict(n_estimators=500, min_samples_leaf=30, max_features=0.5, n_jobs=6, random_state=0); p.update(kw)
    return ExtraTreesClassifier(**p).fit(XA, a.fraud).predict_proba(XB)[:, 1]


def rf(a, b, **kw):
    XA, XB = tree_X(a, b)
    p = dict(n_estimators=500, min_samples_leaf=20, max_features=0.33, n_jobs=6, random_state=0); p.update(kw)
    return RandomForestClassifier(**p).fit(XA, a.fraud).predict_proba(XB)[:, 1]


def hgb(a, b):
    XA, XB = tree_X(a, b)
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=300, min_samples_leaf=50, l2_regularization=5,
                                          random_state=0).fit(XA, a.fraud).predict_proba(XB)[:, 1]


def v4fp(a, b):
    return r6lab.fit_predict(a, b)


def inner_oof(a, k=5, seed=0):
    o = np.zeros(len(a))
    for i, j in StratifiedKFold(k, shuffle=True, random_state=seed).split(a, a.fraud):
        o[j] = v4fp(a.iloc[i], a.iloc[j])
    return o


def hardneg(top=0.05):
    """Specialist: fraud vs legit rows that v4 (inner OOF) ranks in its top `top` share. LR on v4 features. Applied to all rows."""
    def fp(a, b):
        o = inner_oof(a); y = a.fraud.values
        thr = np.quantile(o, 1 - top); m = (y == 1) | (o >= thr)
        K = fit_constants(a); A = make_features(a, K); B = make_features(b, K)
        mdl = models.lr_pipeline(0.1, models.LR_NUM_V4, models.LR_CAT).fit(A[m][models.LR_CAT + models.LR_NUM_V4], y[m])
        return mdl.predict_proba(B[models.LR_CAT + models.LR_NUM_V4])[:, 1]
    return fp


def band_specialist(lo=0.02, hi=0.40):
    """Two-stage: HGB trained only on rows whose inner-OOF v4 rank is in the [lo, hi] top band, inputs = features + v4 logit."""
    def fp(a, b):
        o = inner_oof(a); y = a.fraud.values; base_b = v4fp(a, b)
        r = pd.Series(o).rank(ascending=False, pct=True).values; m = (r >= lo) & (r <= hi)
        XA, XB = tree_X(a, b); XA["v4"] = np.log(o / (1 - o)); XB["v4"] = np.log(base_b / (1 - base_b))
        g = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, max_iter=200, min_samples_leaf=40, l2_regularization=5, random_state=0)
        return g.fit(XA[m], y[m]).predict_proba(XB)[:, 1]
    return fp


def subtype(kind):
    """One-vs-legit detector for one fraud subtype (subtype defined from features only); other frauds dropped from training."""
    def member(D):
        if kind == "device": return D.new_device.fillna(0).values == 1
        if kind == "burst": return D.burst.fillna(0).values == 1
        if kind == "young": return (D.account_age < 365).values
        if kind == "foreign": return D.foreign.fillna(0).values == 1
        if kind == "ordinary": return ~((D.new_device.fillna(0) == 1) | (D.burst.fillna(0) == 1) | (D.account_age < 365) | (D.foreign.fillna(0) == 1)).values
    def fp(a, b):
        K = fit_constants(a); A = make_features(a, K); B = make_features(b, K); y = a.fraud.values
        m = (y == 0) | member(A)
        mdl = models.lr_pipeline(0.1, models.LR_NUM_V4, models.LR_CAT).fit(A[m][models.LR_CAT + models.LR_NUM_V4], y[m])
        return mdl.predict_proba(B[models.LR_CAT + models.LR_NUM_V4])[:, 1]
    return fp


if __name__ == "__main__":
    which = sys.argv[1:] or ["et", "rf", "hgb", "hn5", "hn2", "hn10", "band", "st_device", "st_burst", "st_young", "st_foreign", "st_ordinary"]
    G = {"et": et, "rf": rf, "hgb": hgb, "hn5": hardneg(0.05), "hn2": hardneg(0.02), "hn10": hardneg(0.10), "band": band_specialist(),
         "st_device": subtype("device"), "st_burst": subtype("burst"), "st_young": subtype("young"), "st_foreign": subtype("foreign"), "st_ordinary": subtype("ordinary")}
    for n in which:
        evaluate("base_" + n, G[n], quick=True, ref="v4")
