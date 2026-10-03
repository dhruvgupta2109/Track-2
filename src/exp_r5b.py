import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from evaluate import evaluate
import r5lab
from features_prod import make_features; from constants import fit_constants
from common import CATS
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))

print("=== C sweep (v4 features) ===", flush=True)
for C in [0.03, 0.05, 0.1, 0.2, 0.3, 1, 3, 10, 30]:
    evaluate(f"v4_C{C}", r5lab.F(C=C), quick=True, ref="v4")

def inner_oof(a, k=5, seed=0):
    o = np.zeros(len(a))
    for i, j in StratifiedKFold(k, shuffle=True, random_state=seed).split(a, a.fraud):
        o[j] = r5lab.fit_predict(a.iloc[i], a.iloc[j])
    return o

print("=== OOF-safe hard-example reweighting ===", flush=True)
def hardw(f, top_legit=0.05, low_fraud=0.5):
    def fp(a, b):
        o = inner_oof(a); y = a.fraud.values
        w = np.ones(len(a))
        thr = np.quantile(o[y == 0], 1 - top_legit); w[(y == 0) & (o >= thr)] = f
        thrf = np.quantile(o[y == 1], low_fraud); w[(y == 1) & (o <= thrf)] = f
        return r5lab.fit_predict(a, b, sample_weight=w)
    return fp
for f in [1.25, 1.5, 2.0]:
    evaluate(f"v4_hardw{f}", hardw(f), quick=True, ref="v4")
evaluate("v4_hardw1.5_legitonly", hardw(1.5, low_fraud=0.0), quick=True, ref="v4")

print("=== residual booster on v4 margin (LightGBM init_score, depth<=2) ===", flush=True)
RC = ["transaction_amount", "transaction_hour", "transactions_last_24h", "transactions_last_1h", "account_age", "new_device", "amt_rel_cat", "amt_rel_avg"] + CATS
def resid(n_est=100, lr=0.03, leaves=3, mcs=100, region=None):
    def fp(a, b):
        o = inner_oof(a); base_b = r5lab.fit_predict(a, b)
        K = fit_constants(a); A = make_features(a, K)[RC]; B = make_features(b, K)[RC]
        for c in CATS:
            cats = sorted(A[c].dropna().unique()); A[c] = pd.Categorical(A[c], cats); B[c] = pd.Categorical(B[c], cats)
        m = np.ones(len(a), bool)
        if region: m = o >= np.quantile(o, 1 - region)       # learn corrections only in the top `region` share (ambiguous/high zone)
        g = lgb.LGBMClassifier(n_estimators=n_est, learning_rate=lr, num_leaves=leaves, max_depth=2, min_child_samples=mcs, reg_lambda=10,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=6)
        g.fit(A[m], a.fraud.values[m], init_score=lg(o)[m])
        corr = g.predict(B, raw_score=True)
        zb = lg(base_b)
        if region:
            inreg = base_b >= np.quantile(o, 1 - region); zb = np.where(inreg, zb + corr, zb)
        else: zb = zb + corr
        return 1 / (1 + np.exp(-zb))
    return fp
evaluate("v4_resid_n50", resid(50), quick=True, ref="v4")
evaluate("v4_resid_n150", resid(150), quick=True, ref="v4")
evaluate("v4_resid_top30_n100", resid(100, region=0.3), quick=True, ref="v4")
