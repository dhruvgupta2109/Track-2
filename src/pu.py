"""Positive-unlabeled learning: positives = labelled TRAIN frauds, negatives = the unlabeled TARGET rows (+ alpha * train legit).
If P(x|fraud) is unchanged in the target, the PU odds rank target rows by target-regime fraud probability."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
import models
from r9lab import base, embed, class_dists, EMB
from common import RAW

def mask_like(df, rate, rng):
    df = df.copy()
    for c in RAW:
        if c in ("transaction_amount", "transaction_hour"): continue
        df.loc[rng.random(len(df)) < rate, c] = np.nan
    return df

def pu_fit_predict(a, b, alpha=0.0, C=0.1, n_mask=0, rate=0.017, knn=True):
    """a = labelled training rows (only frauds used as positives unless alpha>0), b = unlabeled target rows."""
    rng = np.random.default_rng(0)
    pos = a[a.fraud == 1]
    if n_mask: pos = pd.concat([pos] + [mask_like(pos, rate, rng) for _ in range(n_mask)])
    parts = [pos.assign(_y=1, _w=1.0 / (1 + n_mask)), b.assign(_y=0, _w=1.0)]
    if alpha > 0: parts.append(a[a.fraud == 0].assign(_y=0, _w=alpha))
    S = pd.concat(parts, ignore_index=True)
    A, B, ya, num = base(a, S)                   # constants / ctx medians from labelled train; features for the stacked set S
    if knn:
        Ea, Es = embed(A, B, EMB); fa, fs = class_dists(Ea, Es, ya, 1, 1); la, ls = class_dists(Ea, Es, ya, 0, 1)
        # for the positive copies of train frauds, nearest-fraud distance must exclude self: use 2nd nearest
        from sklearn.neighbors import NearestNeighbors
        nnf = NearestNeighbors(n_neighbors=2).fit(Ea[ya == 1]); d2 = nnf.kneighbors(Es)[0]
        dfx = np.where(S._y.values == 1, d2[:, 1], d2[:, 0])
        nnl = NearestNeighbors(n_neighbors=2).fit(Ea[ya == 0]); l2 = nnl.kneighbors(Es)[0]
        own_legit = np.r_[np.zeros(len(pos) + len(b), bool), np.ones(len(S) - len(pos) - len(b), bool)]
        dlx = np.where(own_legit, l2[:, 1], l2[:, 0])
        B["dr"] = np.log1p(dfx) - np.log1p(dlx); num = num + ["dr"]
    m = models.lr_pipeline(C, num, models.LR_CAT)
    m.fit(B[models.LR_CAT + num], S._y.values, logisticregression__sample_weight=S._w.values)
    nb = len(b); start = len(pos)
    return m.predict_proba(B.iloc[start:start + nb][models.LR_CAT + num])[:, 1]

if __name__ == "__main__":
    from r9lab import screen
    for alpha in [0.0, 0.1, 0.3, 1.0]:
        screen(f"PU iid (unlabeled = val fold) alpha={alpha}", lambda a, b, al=alpha: pu_fit_predict(a, b, al))
