import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); from common import *; import models
from features_prod import make_features; from constants import fit_constants
from sklearn.metrics import average_precision_score as ap
tr, te = load(); y = tr[LABEL].values; w = np.load("experiments/adv_weights.npy")
def lr_f(num, cat=models.LR_CAT, C=0.1, extra=None):
    def fp(a, b):
        K = fit_constants(a); A = make_features(a, K); B = make_features(b, K)
        if extra: A = extra(A); B = extra(B)
        m = models.lr_pipeline(C, num, cat).fit(A[cat + num], a[LABEL]); return m.predict_proba(B[cat + num])[:, 1]
    return fp
def ex(d):
    d["t1_ge4_nb"] = ((d.transactions_last_1h >= 4) & ~(d.burst == 1)).astype(float).where(d.transactions_last_1h.notna())
    d["t1_hi_nb"] = d.t1_hi * (1 - d.burst.fillna(0)); return d
base = [f for f in models.LR_NUM if f not in ("transactions_last_1h", "t1_hi", "t1_share")]
V = {"full": models.LR_NUM, "no1h": base, "no1h+t1_hi": base + ["t1_hi"], "no1h+raw": base + ["transactions_last_1h"],
     "no1h+t1_hi_nb": base + ["t1_hi_nb"], "no1h+t1_ge4_nb": base + ["t1_ge4_nb"],
     "no1h-spend-foreign": [f for f in base if f not in ("log_spend", "foreign")]}
for seed in [11, 2026]:
    splits = cv_splits(y, n_repeats=5, seed=seed)
    for n, num in V.items():
        oof, s = run_cv(lr_f(num, extra=ex), tr, splits, 5, verbose=False)
        twa = np.mean([ap(y, oof[r], sample_weight=w) for r in range(5)])
        print(f"seed{seed} {n:22s} {s['pr_auc_mean']:.4f} tw {twa:.4f}", flush=True)
        if seed == 2026: np.save(f"experiments/oof_lrv_{n}.npy", oof); s["pr_auc_testweighted"] = twa; log_result("lrv_" + n, s, "1h velocity encoding study")
