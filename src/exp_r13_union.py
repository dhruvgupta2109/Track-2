import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
from scipy.stats import rankdata
import suraj_features as SF, models
from r9lab import base, embed, class_dists, EMB, fit_lr, screen, TR, Y, W_HN, O5, SPLITS
from common import cv_splits, run_cv
SNUM = [c for c in SF.make_features(TR.head(5)).columns if c not in SF.CAT_COLS]
FLAGS = ["high_amount", "burst_1h", "burst_24h", "high_spend_24h", "risky_cat", "risky_cat_cnp", "risky_cat_high_amount", "nd_foreign", "nd_high_amount", "nd_young", "young_account", "account_lt_1y"]
def kdr(A, B, y):
    Ea, Eb = embed(A, B, EMB); fa, fb = class_dists(Ea, Eb, y, 1, 1); la, lb = class_dists(Ea, Eb, y, 0, 1)
    return np.log1p(fa[:, 0]) - np.log1p(la[:, 0]), np.log1p(fb[:, 0]) - np.log1p(lb[:, 0])
def suraj(knn=False, C=0.03):
    def fp(a, b):
        A, B = SF.make_features(a), SF.make_features(b); num = list(SNUM)
        if knn:
            A2, B2, y, _ = base(a, b); A["dr"], B["dr"] = kdr(A2, B2, y); num.append("dr")
        return models.lr_pipeline(C, num, SF.CAT_COLS).fit(A[SF.CAT_COLS + num], a.fraud).predict_proba(B[SF.CAT_COLS + num])[:, 1]
    return fp
def v5_plus_flags(flags=FLAGS, C=0.1):
    def fp(a, b):
        A, B, y, num = base(a, b); A["dr"], B["dr"] = kdr(A, B, y)
        Sa, Sb = SF.make_features(a), SF.make_features(b)
        for f in flags: A["s_" + f] = Sa[f].values; B["s_" + f] = Sb[f].values
        return fit_lr(A, B, y, num + ["dr"] + ["s_" + f for f in flags], C)
    return fp
if __name__ == "__main__":
    oS, *_ = screen("suraj v1 (their features, C=0.03)", suraj())
    oSK, *_ = screen("suraj v1 + my kNN dr", suraj(knn=True))
    oVF, *_ = screen("v5 + their 12 flags", v5_plus_flags())
    screen("v5 + their flags C=0.05", v5_plus_flags(C=0.05))
    screen("v5 + high_amount,risky_cat_cnp,risky_cat_high_amount", v5_plus_flags(["high_amount", "risky_cat_cnp", "risky_cat_high_amount"]))
    rk = lambda x: np.vstack([rankdata(r) / len(r) for r in x]); f = lambda o, w=None: np.mean([ap(Y, o[r], sample_weight=w) for r in range(2)])
    b5 = (f(O5[:2]), f(O5[:2], W_HN))
    for nm, o in [("suraj v1", oS), ("suraj+kNN", oSK), ("v5+flags", oVF)]:
        for w in [0.3, 0.5]:
            q = (1 - w) * rk(O5[:2]) + w * rk(o); print(f"rank blend v5 {1-w} + {nm} {w}: d_std {f(q)-b5[0]:+.4f} d_hnW {f(q, W_HN)-b5[1]:+.4f}", flush=True)
