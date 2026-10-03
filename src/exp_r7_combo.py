import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
from evaluate import evaluate, prevalence_deltas, Y, W_HN, TR
from common import cv_splits, run_cv
import models, r6lab
from features_prod import make_features; from constants import fit_constants

def combo(age=True, ctx=True, knn=True, C=0.1):
    def fp(a, b):
        K = fit_constants(a); A = make_features(a, K); B = make_features(b, K); ya = a.fraud.values
        num = list(models.LR_NUM_V4)
        if age:
            for D in (A, B): D["age_lt120"] = (D.account_age < 120).astype(float).where(D.account_age.notna())
            num.append("age_lt120")
        if ctx:
            ea, eb = r6lab.ctx_amount(A, B, ya); A[ea.columns] = ea; B[eb.columns] = eb; num += list(ea.columns)
        if knn:
            ea, eb = r6lab.knn_feats(A, B, ya); A["d_ratio"] = ea.d_ratio; B["d_ratio"] = eb.d_ratio; num.append("d_ratio")
        return models.lr_pipeline(C, num, models.LR_CAT).fit(A[models.LR_CAT + num], ya).predict_proba(B[models.LR_CAT + num])[:, 1]
    return fp
ref = np.load("experiments/oof5_v4.npy")
o, v = evaluate("r7_combo_age_ctx_knn", combo(), quick=False, ref="v4")
print("   prev deltas:", {k: round(x, 4) for k, x in prevalence_deltas(o, ref).items()})
o2, v2 = evaluate("r7_combo_age_ctx", combo(knn=False), quick=False, ref="v4")
print("   prev deltas:", {k: round(x, 4) for k, x in prevalence_deltas(o2, ref).items()})
for seed in [99, 314, 7]:
    sp = cv_splits(Y, n_repeats=3, seed=seed); res = {}
    for n, fp in [("v4", combo(False, False, False)), ("age_ctx_knn", combo()), ("age_ctx", combo(knn=False))]:
        res[n], _ = run_cv(fp, TR, sp, 3, verbose=False)
    for n in ["age_ctx_knn", "age_ctx"]:
        print(f"fresh seed {seed}: {n}-v4 d_std {np.mean([ap(Y,res[n][r])-ap(Y,res['v4'][r]) for r in range(3)]):+.4f} d_hnW {np.mean([ap(Y,res[n][r],sample_weight=W_HN)-ap(Y,res['v4'][r],sample_weight=W_HN) for r in range(3)]):+.4f}", flush=True)
