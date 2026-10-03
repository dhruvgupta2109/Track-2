import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from r9lab import *
def dr_of(Ea, Eb, y):
    fa, fb = class_dists(Ea, Eb, y, 1, 1); la, lb = class_dists(Ea, Eb, y, 0, 1)
    return np.log1p(fa[:, 0]) - np.log1p(la[:, 0]), np.log1p(fb[:, 0]) - np.log1p(lb[:, 0])
def rsub(n=30, size=6, keep=True, seed=0):
    def fn(A, B, y):
        Ea, Eb = embed(A, B, EMB); rng = np.random.default_rng(seed); ra, rb = [], []
        for _ in range(n):
            c = rng.choice(Ea.shape[1], size, replace=False); a_, b_ = dr_of(Ea[:, c], Eb[:, c], y); ra.append(a_); rb.append(b_)
        oa, ob = pd.DataFrame({"dr_rs": np.mean(ra, 0)}), pd.DataFrame({"dr_rs": np.mean(rb, 0)})
        if keep: oa["dr"], ob["dr"] = dr_of(Ea, Eb, y)
        return oa, ob
    return fn
def rbag(n=20, frac=0.7, seed=0):
    def fn(A, B, y):
        Ea, Eb = embed(A, B, EMB); rng = np.random.default_rng(seed); ra, rb = [], []
        from sklearn.neighbors import NearestNeighbors
        for _ in range(n):
            s = rng.random(len(y)) < frac; out = []
            for lab in (1, 0):
                R = Ea[s & (y == lab)]; nn = NearestNeighbors(n_neighbors=2).fit(R); d = nn.kneighbors(Ea)[0]
                own = s & (y == lab); out.append((np.where(own, d[:, 1], d[:, 0]), nn.kneighbors(Eb, n_neighbors=1)[0][:, 0]))
            ra.append(np.log1p(out[0][0]) - np.log1p(out[1][0])); rb.append(np.log1p(out[0][1]) - np.log1p(out[1][1]))
        oa, ob = pd.DataFrame({"dr_bag": np.mean(ra, 0)}), pd.DataFrame({"dr_bag": np.mean(rb, 0)})
        oa["dr"], ob["dr"] = dr_of(Ea, Eb, y); return oa, ob
    return fn
screen("v5 + random-subspace dr (30 x 6 dims)", make(rsub()))
screen("random-subspace dr only (replace)", make(rsub(keep=False)))
screen("v5 + random-subspace dr (50 x 4 dims)", make(rsub(50, 4)))
screen("v5 + random-subspace dr (30 x 8 dims)", make(rsub(30, 8)))
screen("v5 + bagged-reference dr (20 x 70%)", make(rbag()))
