import sys, warnings, json; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from r9lab import *
def prep(D):
    D["cash"] = (D.merchant_category == "cash_transfer").astype(float); return D
def local(cols, w=None, pool=None):
    def fn(A, B, y):
        prep(A); prep(B); Ea, Eb = embed(A, B, cols, w)
        if pool:
            ga, gb = A[pool].values, B[pool].values
            fa, fb = pooled_dists(Ea, Eb, ga, gb, y, 1, 1); la, lb = pooled_dists(Ea, Eb, ga, gb, y, 0, 1)
        else:
            fa, fb = class_dists(Ea, Eb, y, 1, 1); la, lb = class_dists(Ea, Eb, y, 0, 1)
        return pd.DataFrame({"dr": np.log1p(fa[:, 0]) - np.log1p(la[:, 0])}), pd.DataFrame({"dr": np.log1p(fb[:, 0]) - np.log1p(lb[:, 0])})
    return fn
E1 = [c for c in EMB if c != "amt_rel_avg"] + ["burst"]
E2 = E1 + ["cash"]
screen("E1 = -amt_rel_avg +burst", make(local(E1)))
screen("E2 = E1 + cash", make(local(E2)))
screen("E1 + merchant pool", make(local(E1, pool="merchant_category")))
screen("E2 + merchant pool", make(local(E2, pool="merchant_category")))
# coordinate search on diagonal weights over E1 (multipliers), 2 passes, accept if std+hnW improves
cols = E1; w = np.ones(len(cols)); best = screen("CS start E1", make(local(cols, w)), log=False)
score = lambda r: r[1] + r[2]
bs = score(best)
for p in range(2):
    for j, c in enumerate(cols):
        for m in [0.5, 2.0]:
            w2 = w.copy(); w2[j] *= m
            r = screen(f"CS p{p} {c} x{m}", make(local(cols, w2)), log=False)
            if score(r) > bs + 0.0005: w, bs = w2, score(r); print("   accepted ->", dict(zip(cols, np.round(w, 2))), flush=True)
json.dump(dict(zip(cols, w.tolist())), open("experiments/r9_metric_weights.json", "w"))
print("final weights", dict(zip(cols, np.round(w, 2))))
screen("CS final weights", make(local(cols, w)))
