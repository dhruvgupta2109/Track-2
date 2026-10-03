"""Fold-safe synthetic augmentation for the v5 pipeline. Generators only ever see the TRAINING fold.
kNN reference sets stay real-only; synthetic rows get distances to real references (2nd-nearest fraud for synthetic frauds,
so an interpolated point is not trivially next to its own parent)."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import models, r6lab
from features_prod import make_features
from constants import fit_constants
from r9lab import embed, class_dists, EMB
from common import RAW, CATS
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm

LOGC = ["transaction_amount", "spend_last_24h", "account_age"]
CNT = ["transactions_last_24h", "transactions_last_1h"]
NUMC = LOGC + CNT
TEST_MISS = 0.022


def _num_space(df):
    X = np.c_[np.log1p(df[LOGC].values), df[CNT].values].astype(float)
    med = np.nanmedian(X, 0); X = np.where(np.isnan(X), med, X)
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def _finish(S, rng, miss=TEST_MISS):
    S["transactions_last_24h"] = np.clip(np.round(S.transactions_last_24h), 1, 27)
    S["transactions_last_1h"] = np.clip(np.minimum(np.round(S.transactions_last_1h), S.transactions_last_24h), 0, 9)
    S["account_age"] = np.clip(np.round(S.account_age), 5, 4000); S["transaction_amount"] = np.clip(S.transaction_amount, 2.5, 9000).round(2)
    S["spend_last_24h"] = np.clip(S.spend_last_24h, 0, None).round(2)
    if miss:
        for c in RAW:
            if c in ("transaction_amount", "transaction_hour"): continue
            S.loc[rng.random(len(S)) < miss, c] = np.nan
    return S


def smote(a, n, rng, k=5, pool=None, label=1):
    """interpolate numerics between an anchor and one of its k nearest same-label rows; categoricals/hour/device from the anchor."""
    P = a[(a.fraud == label) & (pool if pool is not None else True)]
    if len(P) < k + 1: return P.iloc[:0]
    Z = _num_space(P); nn = NearestNeighbors(n_neighbors=k + 1).fit(Z); _, J = nn.kneighbors(Z)
    i = rng.integers(0, len(P), n); j = J[i, rng.integers(1, k + 1, n)]; lam = rng.random(n)[:, None]
    A_, B_ = P.iloc[i].reset_index(drop=True), P.iloc[j].reset_index(drop=True); S = A_.copy()
    xa = np.c_[np.log1p(A_[LOGC].values), A_[CNT].values]; xb = np.c_[np.log1p(B_[LOGC].values), B_[CNT].values]
    xb = np.where(np.isnan(xb), xa, xb); x = xa + lam * (xb - xa)
    S[LOGC] = np.expm1(x[:, :3]); S[CNT] = x[:, 3:]
    return _finish(S, rng)


def jitter(a, n, rng, s=0.15, label=1):
    P = a[a.fraud == label]; S = P.iloc[rng.integers(0, len(P), n)].reset_index(drop=True).copy()
    for c in LOGC: S[c] = S[c] * np.exp(rng.normal(0, s, n))
    for c in CNT: S[c] = S[c] + rng.choice([-1, 0, 0, 0, 1], n)
    return _finish(S, rng)


def copula(a, n, rng, label=1):
    """Gaussian copula on fraud numerics (normal scores of ECDF); categoricals/hour/device copied from the nearest real fraud."""
    P = a[a.fraud == label].reset_index(drop=True); X = np.c_[np.log1p(P[LOGC].values), P[CNT].values]
    X = np.where(np.isnan(X), np.nanmedian(X, 0), X); m = len(P)
    U = (np.argsort(np.argsort(X, 0), 0) + 0.5) / m; G = norm.ppf(U); C = np.corrcoef(G.T) + 1e-3 * np.eye(X.shape[1])
    g = rng.multivariate_normal(np.zeros(X.shape[1]), C, n); u = norm.cdf(g)
    Xs = np.column_stack([np.quantile(X[:, j], u[:, j]) for j in range(X.shape[1])])
    Zr = (X - X.mean(0)) / (X.std(0) + 1e-9); Zs = (Xs - X.mean(0)) / (X.std(0) + 1e-9)
    nn = NearestNeighbors(n_neighbors=1).fit(Zr); j = nn.kneighbors(Zs, return_distance=False)[:, 0]
    S = P.iloc[j].reset_index(drop=True).copy(); S[LOGC] = np.expm1(Xs[:, :3]); S[CNT] = Xs[:, 3:]
    return _finish(S, rng)


def ordinary_pool(a):
    return (a.new_device.fillna(0) == 0) & (a.transactions_last_24h.fillna(0) < 16) & (a.account_age.fillna(1000) >= 365) & (a.country == "SG")


def risky_legit_pool(a):
    return (a.new_device.fillna(0) == 1) | (a.transactions_last_24h.fillna(0) >= 16) | (a.account_age.fillna(1000) < 90)


def fit_predict(a, b, syn=None, pseudo=None):
    """v5 pipeline on real train fold + synthetic rows (with column 'w'); pseudo = (DataFrame rows from b, labels, weight)."""
    K = fit_constants(a); ya = a.fraud.values
    parts = [a.assign(w=1.0)]
    if syn is not None and len(syn): parts.append(syn)
    if pseudo is not None: parts.append(pseudo)
    full = pd.concat(parts, ignore_index=True); nreal = len(a)
    Ar, Af, B = make_features(a, K), make_features(full, K), make_features(b, K)
    num = list(models.LR_NUM_V4)
    for D in (Af, B): D["age_lt120"] = (D.account_age < 120).astype(float).where(D.account_age.notna())
    num.append("age_lt120")
    _, cf = r6lab.ctx_amount(Ar, Af, ya); _, cb = r6lab.ctx_amount(Ar, B, ya)
    Af[cf.columns] = cf.values; B[cb.columns] = cb.values; num += list(cf.columns)
    Er, Ef = embed(Ar, Af, EMB); _, Eb = embed(Ar, B, EMB)
    dfr, dfx = class_dists(Er, np.r_[Ef[nreal:], Eb], ya, 1, 2); dlr, dlx = class_dists(Er, np.r_[Ef[nreal:], Eb], ya, 0, 1)
    ns = len(full) - nreal; yfull = full.fraud.values
    # synthetic/pseudo rows: fraud-labelled ones use 2nd-nearest real fraud
    dfs = np.where(yfull[nreal:] == 1, dfx[:ns, 1], dfx[:ns, 0]) if ns else np.zeros(0)
    Af["dr"] = np.r_[np.log1p(dfr[:, 0]) - np.log1p(dlr[:, 0]), np.log1p(dfs) - np.log1p(dlx[:ns, 0])]
    B["dr"] = np.log1p(dfx[ns:, 0]) - np.log1p(dlx[ns:, 0]); num.append("dr")
    m = models.lr_pipeline(0.1, num, models.LR_CAT)
    m.fit(Af[models.LR_CAT + num], yfull, logisticregression__sample_weight=full.w.values)
    return m.predict_proba(B[models.LR_CAT + num])[:, 1]
