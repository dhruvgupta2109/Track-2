import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from evaluate import evaluate, prevalence_deltas
import models, r6lab
from features_prod import make_features; from constants import fit_constants
o5 = np.load("experiments/oof5_v5_prod.npy")
BASE = r6lab.EMB
def emb(A, B, cols, cat_w=0.0, hour_w=0.0, drop=()):
    cols = [c for c in cols if c not in drop]; med = A[cols].median(); sc = StandardScaler().fit(A[cols].fillna(med))
    Ea, Eb = sc.transform(A[cols].fillna(med)), sc.transform(B[cols].fillna(med))
    if cat_w:
        for c in ["merchant_category", "transaction_channel", "country"]:
            lv = sorted(A[c].dropna().unique())
            Ea = np.c_[Ea, cat_w * np.c_[tuple((A[c] == v).values for v in lv)]]; Eb = np.c_[Eb, cat_w * np.c_[tuple((B[c] == v).values for v in lv)]]
    if hour_w:
        for E, D in [(0, A), (1, B)]:
            pass
        ha, hb = A.transaction_hour.fillna(12).values, B.transaction_hour.fillna(12).values
        Ea = np.c_[Ea, hour_w * np.cos(2 * np.pi * ha / 24), hour_w * np.sin(2 * np.pi * ha / 24)]
        Eb = np.c_[Eb, hour_w * np.cos(2 * np.pi * hb / 24), hour_w * np.sin(2 * np.pi * hb / 24)]
    return Ea, Eb
def dratio(Ea, Eb, ya):
    out = []
    for E, own in [(Ea, True), (Eb, False)]:
        r = {}
        for lab in (1, 0):
            d = NearestNeighbors(n_neighbors=2).fit(Ea[ya == lab]).kneighbors(E)[0]
            r[lab] = np.log1p(np.where(ya == lab, d[:, 1], d[:, 0]) if own else d[:, 0])
        out.append(r[1] - r[0])
    return out
def v5e(**ek):
    def fp(a, b):
        K = fit_constants(a); A = make_features(a, K); B = make_features(b, K); ya = a.fraud.values; num = list(models.LR_NUM_V4)
        for D in (A, B): D["age_lt120"] = (D.account_age < 120).astype(float).where(D.account_age.notna())
        num.append("age_lt120"); ea, eb = r6lab.ctx_amount(A, B, ya); A[ea.columns] = ea; B[eb.columns] = eb; num += list(ea.columns)
        Ea, Eb = emb(A, B, BASE + ek.pop("add", []) if False else BASE + ek.get("add", []), ek.get("cat_w", 0), ek.get("hour_w", 0), ek.get("drop", ()))
        A["dr"], B["dr"] = dratio(Ea, Eb, ya); num.append("dr")
        return models.lr_pipeline(0.1, num, models.LR_CAT).fit(A[models.LR_CAT + num], ya).predict_proba(B[models.LR_CAT + num])[:, 1]
    return fp
def rep(name, fp):
    o, v = evaluate(name, fp, quick=True, ref="v5_prod")
    print("    prev deltas vs v5:", {k: round(x, 4) for k, x in prevalence_deltas(o, o5, prevs=(0.0075, 0.01, 0.0177), n=60).items()}, flush=True)
rep("v5_emb_cat0.5", v5e(cat_w=0.5))
rep("v5_emb_cat1", v5e(cat_w=1.0))
rep("v5_emb_hour1", v5e(hour_w=1.0))
rep("v5_emb_cat0.5_hour0.5", v5e(cat_w=0.5, hour_w=0.5))
rep("v5_emb_+spend+young", v5e(add=["log_spend", "young", "burst"]))
rep("v5_emb_-avg", v5e(drop=("avg_spend", "amt_rel_avg")))
