"""Transductive legit references: bottom-q of the unlabeled batch (by a first-pass v5 score) join the legit reference set."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from r9lab import *

def trans_fp(q=0.5, train_too=True):
    def fp(a, b):
        A, B, ya, num = base(a, b)
        Ea, Eb = embed(A, B, EMB)
        fa, fb = class_dists(Ea, Eb, ya, 1, 1); la, lb = class_dists(Ea, Eb, ya, 0, 1)
        A["dr"] = np.log1p(fa[:, 0]) - np.log1p(la[:, 0]); B["dr"] = np.log1p(fb[:, 0]) - np.log1p(lb[:, 0])
        p1 = fit_lr(A, B, ya, num + ["dr"])                              # first pass (= v5)
        if q <= 0: return p1
        P = Eb[p1 <= np.quantile(p1, q)]                                  # pseudo-legit unlabeled rows
        pidx = np.where(p1 <= np.quantile(p1, q))[0]
        nn = NearestNeighbors(n_neighbors=2).fit(P)
        d, j = nn.kneighbors(Eb); self_hit = pidx[j[:, 0]] == np.arange(len(Eb))
        dpb = np.where(self_hit, d[:, 1], d[:, 0])                        # exclude self
        lb2 = np.minimum(lb[:, 0], dpb); B["dr"] = np.log1p(fb[:, 0]) - np.log1p(lb2)
        if train_too:
            dpa = nn.kneighbors(Ea, n_neighbors=1)[0][:, 0]; la2 = np.minimum(la[:, 0], dpa); A["dr"] = np.log1p(fa[:, 0]) - np.log1p(la2)
        return fit_lr(A, B, ya, num + ["dr"])
    return fp
screen("trans q=0 (v5)", trans_fp(0))
for q in [0.3, 0.5, 0.7, 0.85]:
    screen(f"trans q={q} train_too", trans_fp(q))
screen("trans q=0.5 valid-only", trans_fp(0.5, False))
