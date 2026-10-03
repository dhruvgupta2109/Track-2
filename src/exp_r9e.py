"""Fraud-subtype kNN (distance to nearest fraud of each mechanism) and multi-metric views, on E1."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from r9lab import *
E1 = [c for c in EMB if c != "amt_rel_avg"] + ["burst"]
def dr_from(Ea, Eb, y):
    fa, fb = class_dists(Ea, Eb, y, 1, 1); la, lb = class_dists(Ea, Eb, y, 0, 1)
    return np.log1p(fa[:, 0]) - np.log1p(la[:, 0]), np.log1p(fb[:, 0]) - np.log1p(lb[:, 0])
def subtype_knn(kinds=("nd", "burst", "young", "foreign", "big")):
    def fn(A, B, y):
        Ea, Eb = embed(A, B, E1); a1, b1 = dr_from(Ea, Eb, y)
        la, lb = class_dists(Ea, Eb, y, 0, 1)
        oa, ob = pd.DataFrame({"dr": a1}), pd.DataFrame({"dr": b1})
        mem = {"nd": A.new_device.fillna(0).values == 1, "burst": A.burst.fillna(0).values == 1, "young": (A.account_age < 120).values,
               "foreign": A.foreign.fillna(0).values == 1, "big": (A.amt_rel_cat > 1).values}
        for k in kinds:
            R = (y == 1) & mem[k]; nn = NearestNeighbors(n_neighbors=2).fit(Ea[R])
            da = nn.kneighbors(Ea)[0]; da = np.where(R, da[:, 1], da[:, 0]); db = nn.kneighbors(Eb, n_neighbors=1)[0][:, 0]
            oa["drs_" + k] = np.log1p(da) - np.log1p(la[:, 0]); ob["drs_" + k] = np.log1p(db) - np.log1p(lb[:, 0])
        return oa, ob
    return fn
def multimetric(views):
    def fn(A, B, y):
        oa, ob = pd.DataFrame(), pd.DataFrame()
        for nm, cols in views.items():
            a1, b1 = dr_from(*embed(A, B, cols), y); oa["dr_" + nm] = a1; ob["dr_" + nm] = b1
        return oa, ob
    return fn
screen("E1 + subtype kNN (5)", make(subtype_knn()))
screen("E1 + subtype kNN nd,burst", make(subtype_knn(("nd", "burst"))))
screen("E1 + subtype kNN young,big", make(subtype_knn(("young", "big"))))
V = {"full": E1, "amt_age": ["log_amt", "amt_rel_cat", "log_age", "avg_spend"], "vel_time_dev": ["transactions_last_24h", "transactions_last_1h", "burst", "night", "new_device"]}
screen("multimetric full+amt_age+vel_time_dev", make(multimetric(V)))
screen("multimetric full+vel_time_dev", make(multimetric({k: V[k] for k in ["full", "vel_time_dev"]})))
screen("multimetric full+amt_age", make(multimetric({k: V[k] for k in ["full", "amt_age"]})))
