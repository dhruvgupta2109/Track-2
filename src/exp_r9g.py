import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from r9lab import *
E1 = [c for c in EMB if c != "amt_rel_avg"] + ["burst"]
def loc(cols=E1, hinges=(), lin=True, sq=False, inter=()):
    def fn(A, B, y):
        Ea, Eb = embed(A, B, cols); fa, fb = class_dists(Ea, Eb, y, 1, 1); la, lb = class_dists(Ea, Eb, y, 0, 1)
        ra = np.log1p(fa[:, 0]) - np.log1p(la[:, 0]); rb = np.log1p(fb[:, 0]) - np.log1p(lb[:, 0])
        oa, ob = pd.DataFrame(), pd.DataFrame()
        if lin: oa["dr"], ob["dr"] = ra, rb
        if sq: oa["dr2"], ob["dr2"] = ra ** 2, rb ** 2
        for q in hinges:
            t = np.quantile(ra, q); oa[f"h{q}"] = np.minimum(ra - t, 0); ob[f"h{q}"] = np.minimum(rb - t, 0)
        for c in inter:
            oa["dr_x_" + c] = ra * A[c].fillna(0).values; ob["dr_x_" + c] = rb * B[c].fillna(0).values
        return oa, ob
    return fn
screen("E1 linear", make(loc()))
screen("E1 + hinge q0.10", make(loc(hinges=(0.10,))))
screen("E1 + hinge q0.05,0.15", make(loc(hinges=(0.05, 0.15))))
screen("E1 + hinge q0.20", make(loc(hinges=(0.20,))))
screen("E1 hinge q0.10 only (no linear)", make(loc(hinges=(0.10,), lin=False)))
screen("E1 + square", make(loc(sq=True)))
screen("E1 + dr x new_device", make(loc(inter=("new_device",))))
screen("E1 + dr x burst", make(loc(inter=("burst",))))
screen("E1 + dr x night", make(loc(inter=("night",))))
