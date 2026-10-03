"""Meta-rankers on strict OOF base scores. Meta-level CV over rows (5-fold, different seed per repeat)."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as ap
from evaluate import Y, W_HN, boot_diff, prevalence_deltas

L = lambda n: np.load(f"experiments/oof5_{n}.npy")
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
B = {n: L(n) for n in ["v4", "r6_knn_dratio_full", "base_et", "base_rf", "base_hgb", "base_hn5", "base_hn10", "base_band",
                       "base_st_device", "base_st_young", "base_st_foreign", "base_st_burst", "catboost_d4", "lgb_mono"]}
v4 = B["v4"]


def meta(cols, kind="lr", band=None, C=1.0):
    out = np.zeros_like(v4)
    for r in range(5):
        z = lg(v4[r]); rk = rankdata(-v4[r]) / len(Y)
        X = [z]
        for c in cols:
            f = lg(B[c][r])
            if band is not None: f = f * ((rk >= band[0]) & (rk <= band[1]))   # correction only active inside the score band
            X.append(f)
        X = np.c_[tuple(X)]
        for i, j in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(X, Y):
            if kind == "lr": m = LogisticRegression(C=C, max_iter=2000).fit(X[i], Y[i])
            else: m = HistGradientBoostingClassifier(max_depth=2, max_iter=100, learning_rate=0.05, min_samples_leaf=100, l2_regularization=10).fit(X[i], Y[i])
            out[r, j] = m.predict_proba(X[j])[:, 1]
    return out


def report(name, o):
    s = np.mean([ap(Y, o[r]) for r in range(5)]); h = np.mean([ap(Y, o[r], sample_weight=W_HN) for r in range(5)])
    (lo, _, hi), _ = boot_diff(o.mean(0), v4.mean(0)); pdlt = prevalence_deltas(o, v4, prevs=(0.0075, 0.01, 0.0177), n=60)
    print(f"{name:45s} std {s:.4f} ({s-0.2411:+.4f}) hnW {h:.4f} ({h-0.1970:+.4f}) d_std90 {lo:+.4f}..{hi:+.4f} prev {[round(x,4) for x in pdlt.values()]}", flush=True)
    return o


report("meta v4 only (sanity)", meta([]))
for cols in [["base_et"], ["base_rf"], ["base_hgb"], ["catboost_d4"], ["lgb_mono"], ["r6_knn_dratio_full"], ["base_st_device", "base_st_young", "base_st_foreign", "base_st_burst"],
             ["base_et", "base_hgb", "catboost_d4", "r6_knn_dratio_full", "base_st_young"]]:
    report("meta LR v4 + " + "+".join(c.replace("base_", "") for c in cols), meta(cols))
for band in [(0.0, 0.05), (0.0, 0.10), (0.02, 0.40)]:
    report(f"meta LR v4 + hn5 in band {band}", meta(["base_hn5"], band=band))
    report(f"meta LR v4 + hn10 in band {band}", meta(["base_hn10"], band=band))
    report(f"meta LR v4 + band-specialist in {band}", meta(["base_band"], band=band))
report("meta HGB(depth2) v4 + et+hgb+cb+knn+st_young", meta(["base_et", "base_hgb", "catboost_d4", "r6_knn_dratio_full", "base_st_young"], kind="hgb"))
# rank stacking
o = np.zeros_like(v4)
for r in range(5):
    R = np.c_[tuple(rankdata(B[c][r]) / len(Y) for c in ["v4", "base_et", "catboost_d4", "r6_knn_dratio_full", "base_st_young"])]
    for i, j in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(R, Y):
        o[r, j] = LogisticRegression(max_iter=2000).fit(R[i], Y[i]).predict_proba(R[j])[:, 1]
report("rank-stack LR v4,et,cb,knn,st_young", o)
