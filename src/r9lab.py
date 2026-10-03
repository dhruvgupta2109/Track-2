"""Round 9: local-geometry lab. v5a base features + pluggable fold-safe neighbourhood features.
Fast screen = first 2 repeats of evaluate.SPLITS, paired against v5 OOF on the same folds."""
import sys, os, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, os.path.dirname(__file__))
import models, r6lab
from features_prod import make_features
from constants import fit_constants
from evaluate import TR, Y, SPLITS, W_HN
from common import run_cv

EMB = list(r6lab.EMB)
O5 = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "oof5_v5_prod.npy"))


def base(a, b):
    K = fit_constants(a); A = make_features(a, K); B = make_features(b, K); ya = a.fraud.values
    num = list(models.LR_NUM_V4)
    for D in (A, B): D["age_lt120"] = (D.account_age < 120).astype(float).where(D.account_age.notna())
    num.append("age_lt120")
    ea, eb = r6lab.ctx_amount(A, B, ya); A[ea.columns] = ea; B[eb.columns] = eb; num += list(ea.columns)
    return A, B, ya, num


def embed(A, B, cols=None, w=None):
    cols = cols or EMB; med = A[cols].median(); Xa = A[cols].fillna(med).values; Xb = B[cols].fillna(med).values
    mu, sd = Xa.mean(0), Xa.std(0); sd[sd == 0] = 1
    Ea, Eb = (Xa - mu) / sd, (Xb - mu) / sd
    if w is not None: Ea, Eb = Ea * np.sqrt(w), Eb * np.sqrt(w)
    return Ea, Eb


def class_dists(Ea, Eb, ya, lab, k):
    """distances from every A row (self excluded) and B row to the k nearest reference rows of class `lab`."""
    R = Ea[ya == lab]; k = min(k, len(R) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(R)
    da = nn.kneighbors(Ea)[0]; da = np.where((ya == lab)[:, None], da[:, 1:], da[:, :k])
    db = nn.kneighbors(Eb, n_neighbors=k)[0]
    return da, db


def pooled_dists(Ea, Eb, ga, gb, ya, lab, k, min_ref=30):
    """same as class_dists but restricted to reference rows in the same group (fallback to global)."""
    da_g, db_g = class_dists(Ea, Eb, ya, lab, k)
    da, db = da_g.copy(), db_g.copy()
    for g in np.unique(ga[~pd.isna(ga)]):
        ma, mb = ga == g, gb == g
        if (ya[ma] == lab).sum() < min_ref + 1: continue
        d1, d2 = class_dists(Ea[ma], Eb[mb], ya[ma], lab, k)
        da[ma] = d1; db[mb] = d2
    return da, db


def fit_lr(A, B, ya, num, C=0.1):
    return models.lr_pipeline(C, num, models.LR_CAT).fit(A[models.LR_CAT + num], ya).predict_proba(B[models.LR_CAT + num])[:, 1]


def make(local_fn, C=0.1):
    """local_fn(A, B, ya) -> (DataFrame for A, DataFrame for B) of extra columns."""
    def fp(a, b):
        A, B, ya, num = base(a, b)
        fa, fb = local_fn(A, B, ya)
        for c in fa.columns: A[c] = fa[c].values; B[c] = fb[c].values
        return fit_lr(A, B, ya, num + list(fa.columns), C)
    return fp


def v5_local(A, B, ya, cols=None, w=None, kf=1, kl=1, agg="min"):
    Ea, Eb = embed(A, B, cols, w)
    fa_, fb_ = class_dists(Ea, Eb, ya, 1, kf); la_, lb_ = class_dists(Ea, Eb, ya, 0, kl)
    red = (lambda d: d.mean(1)) if agg == "mean" else (lambda d: d[:, -1] if agg == "kth" else d[:, 0])
    ra = np.log1p(red(fa_)) - np.log1p(red(la_)); rb = np.log1p(red(fb_)) - np.log1p(red(lb_))
    return pd.DataFrame({"dr": ra}), pd.DataFrame({"dr": rb})


def screen(name, fp, reps=2, log=True):
    oof, _ = run_cv(fp, TR, SPLITS[: 5 * reps], reps, verbose=False)
    s = np.mean([ap(Y, oof[r]) for r in range(reps)]); h = np.mean([ap(Y, oof[r], sample_weight=W_HN) for r in range(reps)])
    s5 = np.mean([ap(Y, O5[r]) for r in range(reps)]); h5 = np.mean([ap(Y, O5[r], sample_weight=W_HN) for r in range(reps)])
    # paired fold-level deltas
    fd = [ap(Y[b], oof[i // 5, b]) - ap(Y[b], O5[i // 5, b]) for i, (a, b) in enumerate(SPLITS[: 5 * reps])]
    msg = f"{name:40s} std {s:.4f} (d {s - s5:+.4f})  hnW {h:.4f} (d {h - h5:+.4f})  folds+ {np.mean(np.array(fd) > 0):.1f}"
    print(msg, flush=True)
    if log:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "r9_screen.log"), "a") as f: f.write(msg + "\n")
    return oof, s - s5, h - h5
