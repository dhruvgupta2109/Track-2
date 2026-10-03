import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.kernel_approximation import Nystroem
from sklearn.cluster import KMeans
from r9lab import *
E1 = [c for c in EMB if c != "amt_rel_avg"] + ["burst"]
def dr_from(Ea, Eb, y):
    fa, fb = class_dists(Ea, Eb, y, 1, 1); la, lb = class_dists(Ea, Eb, y, 0, 1)
    return np.log1p(fa[:, 0]) - np.log1p(la[:, 0]), np.log1p(fb[:, 0]) - np.log1p(lb[:, 0])
def nca(dim=6, keep_v5=True, nlegit=3000):
    def fn(A, B, y):
        Ea, Eb = embed(A, B, E1); rng = np.random.default_rng(0)
        idx = np.r_[np.where(y == 1)[0], rng.choice(np.where(y == 0)[0], nlegit, replace=False)]
        T = NeighborhoodComponentsAnalysis(n_components=dim, max_iter=60, random_state=0).fit(Ea[idx], y[idx])
        a2, b2 = dr_from(T.transform(Ea), T.transform(Eb), y)
        out_a, out_b = pd.DataFrame({"dr_nca": a2}), pd.DataFrame({"dr_nca": b2})
        if keep_v5:
            a1, b1 = dr_from(Ea, Eb, y); out_a["dr"] = a1; out_b["dr"] = b1
        return out_a, out_b
    return fn
def nystroem(nc=200, gamma=0.1):
    def fn(A, B, y):
        Ea, Eb = embed(A, B, E1); a1, b1 = dr_from(Ea, Eb, y)
        ny = Nystroem(gamma=gamma, n_components=nc, random_state=0).fit(Ea)
        Za, Zb = ny.transform(Ea), ny.transform(Eb)
        fa = pd.DataFrame(Za, columns=[f"ny{i}" for i in range(nc)]); fb = pd.DataFrame(Zb, columns=fa.columns)
        fa["dr"] = a1; fb["dr"] = b1
        return fa, fb
    return fn
def protos(kf=15, kl=60, sigmas=(0.5, 1.0, 2.0)):
    def fn(A, B, y):
        Ea, Eb = embed(A, B, E1); a1, b1 = dr_from(Ea, Eb, y)
        cf = KMeans(kf, n_init=5, random_state=0).fit(Ea[y == 1]).cluster_centers_; cl = KMeans(kl, n_init=3, random_state=0).fit(Ea[y == 0]).cluster_centers_
        def feats(E, base):
            dF = np.sqrt(((E[:, None, :] - cf[None]) ** 2).sum(-1)); dL = np.sqrt(((E[:, None, :] - cl[None]) ** 2).sum(-1))
            o = pd.DataFrame({"dr": base, "proto_ratio": np.log1p(dF.min(1)) - np.log1p(dL.min(1))})
            for s in sigmas:
                o[f"rbfF{s}"] = np.log(np.exp(-dF ** 2 / (2 * s * s)).sum(1) + 1e-9); o[f"rbfL{s}"] = np.log(np.exp(-dL ** 2 / (2 * s * s)).sum(1) + 1e-9)
            return o
        return feats(Ea, a1), feats(Eb, b1)
    return fn
screen("E1 baseline", make(lambda A, B, y: (lambda r: (pd.DataFrame({"dr": r[0]}), pd.DataFrame({"dr": r[1]})))(dr_from(*embed(A, B, E1), y))))
screen("E1 + NCA dim6 (both)", make(nca(6)))
screen("NCA dim6 only", make(nca(6, keep_v5=False)))
screen("E1 + NCA dim3", make(nca(3)))
screen("E1 + prototypes/RBF", make(protos()))
screen("E1 + Nystroem200 g0.1 C0.1", make(nystroem(200, 0.1)))
screen("E1 + Nystroem200 g0.1 C0.03", make(nystroem(200, 0.1), C=0.03))
