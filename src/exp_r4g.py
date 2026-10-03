import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score as ap
from sklearn.model_selection import StratifiedKFold
from evaluate import evaluate, Y, TR, W_HN
import lrlab
from exp_r4d_models import v4, _nd2
from common import CATS, RAW
from features_prod import make_features; from constants import fit_constants

print("=== Phase 5: OOF target encoding (nested inside each training fold) ===")
def te_fp(replace_onehot):
    def fp(a, b):
        a = a.copy(); b = b.copy(); prior = a.fraud.mean(); m = 50
        for c in CATS:
            enc_a = np.zeros(len(a))
            for i, j in StratifiedKFold(5, shuffle=True, random_state=0).split(a, a.fraud):
                g = a.iloc[i].groupby(a[c].iloc[i].fillna("NA")).fraud.agg(["sum", "count"])
                s = (g["sum"] + m * prior) / (g["count"] + m)
                enc_a[j] = a[c].iloc[j].fillna("NA").map(s).fillna(prior).values
            g = a.groupby(a[c].fillna("NA")).fraud.agg(["sum", "count"]); s = (g["sum"] + m * prior) / (g["count"] + m)
            a[c + "_te"] = np.log(enc_a / (1 - enc_a)); b[c + "_te"] = np.log(b[c].fillna("NA").map(s).fillna(prior) / (1 - b[c].fillna("NA").map(s).fillna(prior)))
        lrlab.add_inter = lambda d, f: _passthrough(d, f)
        cat = [x for x in lrlab.BASE_CAT if not (replace_onehot and x in CATS)]
        try: return lrlab.fit_predict(a, b, fams=("nd",), num=lrlab.BASE_NUM + [c + "_te" for c in CATS], cat=cat)
        finally: lrlab.add_inter = _orig
    return fp
_orig = lrlab.add_inter
def _passthrough(d, f):
    d, cols = _nd2(d, f); return d, cols
evaluate("v4_TE_replace_onehot", te_fp(True), quick=True)
evaluate("v4_TE_plus_onehot", te_fp(False), quick=True)

print("\n=== Phase 6: AP by regime (v3 vs v4 OOF) ===")
o3 = np.load("experiments/oof5_v3_lr.npy"); o4 = np.load("experiments/oof5_v4.npy")
d = TR.copy(); d["young"] = d.account_age < 90; d["burst"] = d.transactions_last_24h >= 16; d["foreign"] = d.country.notna() & (d.country != "SG")
med = d.groupby("merchant_category").transaction_amount.median(); d["big"] = (np.log1p(d.transaction_amount) - np.log1p(d.merchant_category.map(med))) > 1.0
d["nsig"] = d[["young", "burst", "foreign", "big"]].sum(1) + d.new_device.fillna(0)
d["regime"] = np.select([d.nsig >= 2, d.nsig == 1], ["obvious(>=2 signals)", "weak(1 signal)"], "ordinary(0)")
for g, s in d.groupby("regime"):
    m = (d.regime == g).values
    print(f"{g:22s} n={m.sum():6d} fraud={Y[m].sum():4d} rate={Y[m].mean():.4f} share_of_fraud={Y[m].sum()/Y.sum():.2f}  AP v3={np.mean([ap(Y[m], o3[r][m]) for r in range(5)]):.4f} v4={np.mean([ap(Y[m], o4[r][m]) for r in range(5)]):.4f}  lift v4={np.mean([ap(Y[m], o4[r][m]) for r in range(5)])/Y[m].mean():.1f}x")

print("\n=== Phase 13: does importance weighting predict shifted-holdout AP when ONLY covariate shift exists? ===")
tl = np.load("experiments/testlikeness_train.npy"); rng = np.random.default_rng(0); out = []
for rep in range(6):
    sel = rng.random(len(Y)) < np.clip(0.6 * tl / tl.mean() * 0.3, 0, 0.95)        # pseudo-test sampled by test-likeness (covariate shift only)
    D, P = TR[~sel], TR[sel]
    # adversarial D vs P, OOF weights for D rows
    K = fit_constants(D); X = make_features(pd.concat([D, P]), K)[["log_amt", "transaction_hour", "transactions_last_24h", "log_age", "new_device", "amt_rel_cat", "log_spend"] + CATS]
    for c in CATS: X[c] = X[c].astype("category")
    z = np.r_[np.zeros(len(D)), np.ones(len(P))]; pz = np.zeros(len(z))
    for i, j in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, z):
        pz[j] = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50, verbose=-1).fit(X.iloc[i], z[i]).predict_proba(X.iloc[j])[:, 1]
    q = np.clip(pz[:len(D)], .01, .99); w = np.minimum(q / (1 - q), 10)
    yD = D.fraud.values; oofD = np.zeros(len(D))
    for i, j in StratifiedKFold(5, shuffle=True, random_state=rep).split(D, yD): oofD[j] = v4(D.iloc[i], D.iloc[j])
    pP = v4(D, P)
    out.append((ap(yD, oofD), ap(yD, oofD, sample_weight=w), ap(P.fraud, pP), P.fraud.mean()))
o = pd.DataFrame(out, columns=["unweighted_CV", "weighted_CV", "actual_shifted_holdout", "holdout_prev"]); print(o.round(4)); print("mean", o.mean().round(4).to_dict())
print("abs error unweighted %.4f  weighted %.4f" % ((o.unweighted_CV - o.actual_shifted_holdout).abs().mean(), (o.weighted_CV - o.actual_shifted_holdout).abs().mean()))

print("\n=== Phase 15: label ambiguity / noise floor ===")
c = TR.copy(); c["ab"] = pd.qcut(c.transaction_amount, 5, labels=False); c["agb"] = pd.cut(c.account_age, [0, 90, 365, 1500, 5000], labels=False)
c["t24b"] = pd.cut(c.transactions_last_24h, [0, 9, 15, 40], labels=False); c["night"] = c.transaction_hour.isin([23, 0, 1, 2, 3])
cell = ["merchant_category", "transaction_channel", "new_device", "ab", "agb", "t24b", "night"]
g = c.groupby(cell, dropna=False).fraud.agg(["sum", "count"]); g = g[g["sum"] > 0]
mixed = g[g["count"] > g["sum"]]
print("frauds in coarse cells that also contain legit rows: %.3f of all frauds" % (mixed["sum"].sum() / Y.sum()))
cf = c.merge(g.assign(cell_rate=g["sum"] / g["count"]).reset_index()[cell + ["cell_rate"]], on=cell, how="left")
fr = cf[cf.fraud == 1].cell_rate
print("fraud rows by their cell's fraud rate: <2x base %.2f | <5x base %.2f | median cell rate %.3f" % ((fr < 2 * Y.mean()).mean(), (fr < 5 * Y.mean()).mean(), fr.median()))
p4 = o4.mean(0); t = pd.DataFrame({"p": p4, "y": Y})
print("most confident FNs (fraud, lowest OOF p):"); print(TR.assign(p=p4)[Y == 1].nsmallest(8, "p")[RAW + ["p"]].round(3).to_string(index=False))
print("most confident FPs (legit, highest OOF p):"); print(TR.assign(p=p4)[Y == 0].nlargest(8, "p")[RAW + ["p"]].round(3).to_string(index=False))
