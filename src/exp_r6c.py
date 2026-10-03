import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
from evaluate import evaluate, prevalence_deltas, Y, W_HN, TR, TE
from common import cv_splits, run_cv
import r6lab
from features_prod import make_features; from constants import fit_constants
evaluate("r6_knn_dratio_full", r6lab.F(gens=("knn",), keep=["d_ratio"]), quick=False, ref="v4")
for seed in [99, 314, 7]:
    sp = cv_splits(Y, n_repeats=3, seed=seed); res = {}
    for n, fp in [("v4", r6lab.F()), ("dratio", r6lab.F(gens=("knn",), keep=["d_ratio"])), ("knnfull", r6lab.F(gens=("knn",)))]:
        res[n], _ = run_cv(fp, TR, sp, 3, verbose=False)
    for n in ["dratio", "knnfull"]:
        print(f"fresh seed {seed}: {n}-v4 d_std {np.mean([ap(Y,res[n][r])-ap(Y,res['v4'][r]) for r in range(3)]):+.4f} d_hnW {np.mean([ap(Y,res[n][r],sample_weight=W_HN)-ap(Y,res['v4'][r],sample_weight=W_HN) for r in range(3)]):+.4f}", flush=True)
# shift check of d_ratio on real test
K = fit_constants(TR); A = make_features(TR, K); B = make_features(TE, K)
ea, eb = r6lab.knn_feats(A, B, TR.fraud.values)
for c in ea.columns: print(c, "train q10/50/90", np.round(np.quantile(ea[c], [.1, .5, .9]), 3), "test", np.round(np.quantile(eb[c], [.1, .5, .9]), 3))
print("test rows with any missing: d_ratio median", np.median(eb.d_ratio[TE.drop(columns='id').isna().any(axis=1).values]).round(3), "vs complete", np.median(eb.d_ratio[~TE.drop(columns='id').isna().any(axis=1).values]).round(3))
