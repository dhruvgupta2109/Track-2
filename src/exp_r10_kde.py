import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from r9lab import *
def v5dr(Ea, Eb, y):
    fa, fb = class_dists(Ea, Eb, y, 1, 1); la, lb = class_dists(Ea, Eb, y, 0, 1)
    return np.log1p(fa[:, 0]) - np.log1p(la[:, 0]), np.log1p(fb[:, 0]) - np.log1p(lb[:, 0])
def loo_kde(E, R, own_mask, bw):
    """log KDE density of each E row under reference R; rows that are themselves in R get leave-one-out correction."""
    kd = KernelDensity(bandwidth=bw).fit(R); ld = kd.score_samples(E); n = len(R); dim = E.shape[1]
    self_k = -0.5 * dim * np.log(2 * np.pi * bw * bw)              # log kernel at distance 0
    dens = np.exp(ld) * n
    loo = np.where(own_mask, (dens - np.exp(self_k)) / (n - 1), dens / n)
    return np.log(np.clip(loo, 1e-300, None))
def kde_feats(bws=(0.3,), keep_dr=True, legit_only=False):
    def fn(A, B, y):
        Ea, Eb = embed(A, B, EMB); oa, ob = pd.DataFrame(), pd.DataFrame()
        if keep_dr: oa["dr"], ob["dr"] = v5dr(Ea, Eb, y)
        for bw in bws:
            la = loo_kde(Ea, Ea[y == 0], y == 0, bw); lb = loo_kde(Eb, Ea[y == 0], np.zeros(len(Eb), bool), bw)
            if legit_only: oa[f"kl{bw}"], ob[f"kl{bw}"] = la, lb
            else:
                fa = loo_kde(Ea, Ea[y == 1], y == 1, bw); fb = loo_kde(Eb, Ea[y == 1], np.zeros(len(Eb), bool), bw)
                oa[f"kr{bw}"], ob[f"kr{bw}"] = np.clip(fa - la, -50, 50), np.clip(fb - lb, -50, 50)
        return oa, ob
    return fn
def gmm_feats(nf=4, nl=32):
    def fn(A, B, y):
        Ea, Eb = embed(A, B, EMB); oa, ob = pd.DataFrame(), pd.DataFrame(); oa["dr"], ob["dr"] = v5dr(Ea, Eb, y)
        gf = GaussianMixture(nf, covariance_type="full", reg_covar=1e-2, random_state=0).fit(Ea[y == 1])
        gl = GaussianMixture(nl, covariance_type="full", reg_covar=1e-2, random_state=0).fit(Ea[y == 0])
        oa["gmm_r"] = gf.score_samples(Ea) - gl.score_samples(Ea); ob["gmm_r"] = gf.score_samples(Eb) - gl.score_samples(Eb)
        return oa, ob
    return fn
for bw in [0.2, 0.35, 0.5, 0.8]:
    screen(f"v5 + KDE ratio bw{bw}", make(kde_feats((bw,))))
screen("v5 + KDE legit-only bw0.35", make(kde_feats((0.35,), legit_only=True)))
screen("KDE ratio bw0.35 replacing dr", make(kde_feats((0.35,), keep_dr=False)))
screen("v5 + GMM ratio (4 fraud / 32 legit comps)", make(gmm_feats()))
screen("v5 + GMM ratio (2 / 16)", make(gmm_feats(2, 16)))
