import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from evaluate import evaluate, prevalence_deltas
import models, r6lab
from features_prod import make_features; from constants import fit_constants
o5 = np.load("experiments/oof5_v5_prod.npy")
def knn_k(A, B, ya, k=1):
    Ea, Eb = r6lab._emb(A, B); out = []
    for E, own in [(Ea, True), (Eb, False)]:
        r = {}
        for lab, nm in [(1, "f"), (0, "l")]:
            nn = NearestNeighbors(n_neighbors=k + 1).fit(Ea[ya == lab]); d = nn.kneighbors(E)[0]
            if own: d = np.where((ya == lab)[:, None], d[:, 1:], d[:, :k])
            else: d = d[:, :k]
            r[nm] = np.log1p(d.mean(1))
        out.append(r["f"] - r["l"])
    return out
def v5x(k=1, C=0.1, also_k1=False):
    def fp(a, b):
        K = fit_constants(a); A = make_features(a, K); B = make_features(b, K); ya = a.fraud.values
        num = list(models.LR_NUM_V4)
        for D in (A, B): D["age_lt120"] = (D.account_age < 120).astype(float).where(D.account_age.notna())
        num.append("age_lt120")
        ea, eb = r6lab.ctx_amount(A, B, ya); A[ea.columns] = ea; B[eb.columns] = eb; num += list(ea.columns)
        A["dr"], B["dr"] = knn_k(A, B, ya, k); num.append("dr")
        if also_k1 and k != 1: A["dr1"], B["dr1"] = knn_k(A, B, ya, 1); num.append("dr1")
        return models.lr_pipeline(C, num, models.LR_CAT).fit(A[models.LR_CAT + num], ya).predict_proba(B[models.LR_CAT + num])[:, 1]
    return fp
def rep(name, fp):
    o, v = evaluate(name, fp, quick=True, ref="v5_prod")
    print("    prev deltas vs v5:", {k: round(x, 4) for k, x in prevalence_deltas(o, o5, prevs=(0.0075, 0.01, 0.0177), n=60).items()}, flush=True)
for C in [0.05, 0.2, 0.3]: rep(f"v5_C{C}", v5x(1, C))
for k in [3, 5, 10]: rep(f"v5_knnk{k}", v5x(k))
rep("v5_knnk5_plus_k1", v5x(5, also_k1=True))
