"""Round 6: fold-aware feature generators attached to the v4 LR, plus regime experts."""
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import models
from features_prod import make_features
from constants import fit_constants

NUM4, CAT4 = models.LR_NUM_V4, models.LR_CAT
EMB = ["log_amt", "amt_rel_cat", "log_age", "new_device", "transactions_last_24h", "transactions_last_1h", "night", "foreign", "avg_spend", "amt_rel_avg"]
lgt = lambda p: np.log(p / (1 - p))


def _emb(A, B, cols=EMB):
    med = A[cols].median()
    sc = StandardScaler().fit(A[cols].fillna(med))
    return sc.transform(A[cols].fillna(med)), sc.transform(B[cols].fillna(med))


def knn_feats(A, B, ya, k=100, prior_m=20):
    """Local fraud rate among k nearest TRAIN-FOLD rows (self excluded for train rows), smoothed; distances to nearest fraud/legit."""
    Ea, Eb = _emb(A, B); prior = ya.mean()
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Ea)
    ia = nn.kneighbors(Ea, return_distance=False)[:, 1:]
    ib = nn.kneighbors(Eb, n_neighbors=k, return_distance=False)
    fa = (ya[ia].sum(1) + prior_m * prior) / (k + prior_m); fb = (ya[ib].sum(1) + prior_m * prior) / (k + prior_m)
    nf = NearestNeighbors(n_neighbors=2).fit(Ea[ya == 1]); nl = NearestNeighbors(n_neighbors=2).fit(Ea[ya == 0])
    dfa = nf.kneighbors(Ea)[0]; dfa = np.where(ya == 1, dfa[:, 1], dfa[:, 0]); dfb = nf.kneighbors(Eb, n_neighbors=1)[0][:, 0]
    dla = nl.kneighbors(Ea)[0]; dla = np.where(ya == 0, dla[:, 1], dla[:, 0]); dlb = nl.kneighbors(Eb, n_neighbors=1)[0][:, 0]
    oa = pd.DataFrame({"knn_rate": lgt(fa), "d_fraud": np.log1p(dfa), "d_legit": np.log1p(dla)}, index=A.index)
    ob = pd.DataFrame({"knn_rate": lgt(fb), "d_fraud": np.log1p(dfb), "d_legit": np.log1p(dlb)}, index=B.index)
    oa["d_ratio"] = oa.d_fraud - oa.d_legit; ob["d_ratio"] = ob.d_fraud - ob.d_legit
    return oa, ob


def iforest_feats(A, B, ya):
    Ea, Eb = _emb(A, B)
    m = IsolationForest(n_estimators=300, random_state=0, n_jobs=4).fit(Ea[ya == 0])
    return pd.DataFrame({"anom": -m.score_samples(Ea)}, index=A.index), pd.DataFrame({"anom": -m.score_samples(Eb)}, index=B.index)


def legit_density(A, B, ya, k=20):
    """Mean distance to k nearest LEGIT train rows (typicality under legit data); self excluded."""
    Ea, Eb = _emb(A, B); L = Ea[ya == 0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(L)
    da = nn.kneighbors(Ea)[0]; da = np.where((ya == 0)[:, None], da[:, 1:], da[:, :k]).mean(1)
    db = nn.kneighbors(Eb, n_neighbors=k)[0].mean(1)
    return pd.DataFrame({"legit_dist": np.log1p(da)}, index=A.index), pd.DataFrame({"legit_dist": np.log1p(db)}, index=B.index)


def woe_feats(A, B, ya, cols=("log_amt", "amt_rel_cat", "log_age", "transactions_last_24h", "transactions_last_1h", "amt_rel_avg", "avg_spend"), nb=10, m=30):
    """Quantile bins fitted on the train fold; smoothed WoE per bin, out-of-fold inside the training fold."""
    prior = ya.mean(); oa = pd.DataFrame(index=A.index); ob = pd.DataFrame(index=B.index)
    for c in cols:
        edges = np.unique(np.nanquantile(A[c], np.linspace(0, 1, nb + 1)[1:-1]))
        ba = np.digitize(A[c].fillna(A[c].median()), edges); bb = np.digitize(B[c].fillna(A[c].median()), edges)

        def rate(bf, yf, bapp):
            s = pd.Series(yf).groupby(bf).agg(["sum", "count"]); r = (s["sum"] + m * prior) / (s["count"] + m)
            return lgt(pd.Series(bapp).map(r).fillna(prior).values)
        va = np.zeros(len(A))
        for i, j in StratifiedKFold(5, shuffle=True, random_state=1).split(ba, ya):
            va[j] = rate(ba[i], ya[i], ba[j])
        oa["woe_" + c] = va; ob["woe_" + c] = rate(ba, ya, bb)
    return oa, ob


def ctx_amount(A, B, ya):
    oa = pd.DataFrame(index=A.index); ob = pd.DataFrame(index=B.index)
    for g in ["transaction_channel", "country"]:
        med = A.groupby(g).log_amt.median(); gm = A.log_amt.median()
        oa["amt_rel_" + g] = A.log_amt - A[g].map(med).fillna(gm); ob["amt_rel_" + g] = B.log_amt - B[g].map(med).fillna(gm)
    for D, O in [(A, oa), (B, ob)]:
        pc = np.full(len(D), 0.5)
        for cat, s in A.groupby("merchant_category").log_amt:
            v = np.sort(s.dropna().values); msk = (D.merchant_category == cat).values
            pc[msk] = np.searchsorted(v, D.log_amt.values[msk]) / len(v)
        O["amt_pct_cat"] = pc
    return oa, ob


GENS = {"knn": knn_feats, "iforest": iforest_feats, "legitdens": legit_density, "woe": woe_feats, "ctxamt": ctx_amount}


def fit_predict(a, b, gens=(), keep=None, C=0.1, drop=()):
    K = fit_constants(a); A = make_features(a, K); B = make_features(b, K); ya = a.fraud.values
    num = [c for c in NUM4 if c not in drop]
    for g in gens:
        ea, eb = GENS[g](A, B, ya); cols = [c for c in ea.columns if keep is None or c in keep]
        A[cols] = ea[cols]; B[cols] = eb[cols]; num += cols
    m = models.lr_pipeline(C, num, CAT4).fit(A[CAT4 + num], ya)
    return m.predict_proba(B[CAT4 + num])[:, 1]


def F(**kw):
    return lambda a, b: fit_predict(a, b, **kw)


def gate_values(D, gate):
    if gate == "new_device": return D.new_device.fillna(0).values
    if gate == "young": return (D.account_age < 365).astype(float).values
    if gate == "foreign": return D.foreign.fillna(0).values
    if gate == "burst": return D.burst.fillna(0).values
    raise ValueError(gate)


def moe(a, b, gate="new_device", C=0.1):
    """Mixture of experts as one LR on [X, gate*X, gate]: separate coefficients per regime, shared penalty."""
    K = fit_constants(a); A = make_features(a, K); B = make_features(b, K); ya = a.fraud.values
    ga, gb = gate_values(A, gate), gate_values(B, gate)
    ct = ColumnTransformer([("c", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(handle_unknown="ignore", sparse_output=False)), CAT4),
                            ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM4)]).fit(A[CAT4 + NUM4])
    Xa = ct.transform(A[CAT4 + NUM4]); Xb = ct.transform(B[CAT4 + NUM4])
    Xa2 = np.c_[Xa, Xa * ga[:, None], ga]; Xb2 = np.c_[Xb, Xb * gb[:, None], gb]
    return LogisticRegression(C=C, max_iter=5000).fit(Xa2, ya).predict_proba(Xb2)[:, 1]
