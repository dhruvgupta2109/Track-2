"""Where is the current best blend (R2 base: 0.8*(0.7 v5 + 0.3 poly) + 0.2 GALR) still systematically wrong? OOF residual scan.
z = (observed frauds - expected) / sqrt(sum p(1-p)) per bin, averaged over the 5 OOF repeats (blend recalibrated per repeat by Platt)."""
import sys, numpy as np, pandas as pd, itertools
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "src")
from evaluate import TR, Y
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
O5 = np.load("experiments/oof5_v5_prod.npy"); PO = np.load("experiments/oof5_bl_poly.npy"); GA = np.load("experiments/oof5_galr_C0.05_int.npy")
P = []
for r in range(5):
    z = 0.8 * (0.7 * lg(O5[r]) + 0.3 * lg(PO[r])) + 0.2 * lg(GA[r])
    P.append(LogisticRegression(C=1e6).fit(z[:, None], Y).predict_proba(z[:, None])[:, 1])
p = np.mean(P, 0)
d = TR.copy()
d["amt_rel"] = d.transaction_amount / d.groupby("merchant_category").transaction_amount.transform("median")
d["avg_sp"] = d.spend_last_24h / d.transactions_last_24h.clip(lower=1); d["amt_vs_avg"] = d.transaction_amount / (d.avg_sp + 1)
d["t1_share"] = d.transactions_last_1h / d.transactions_last_24h.clip(lower=1)
bins = {"transaction_amount": [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000, 5000, 1e9],
        "amt_rel": [0, .25, .5, .75, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 1e9], "spend_last_24h": [-1, 0.5, 50, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 5000, 1e9],
        "account_age": [-1, 30, 60, 90, 120, 180, 270, 365, 500, 730, 1000, 1500, 2000, 3000, 1e9], "transactions_last_24h": [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 30],
        "transactions_last_1h": [-1, 0, 1, 2, 3, 4, 5, 6, 8, 10], "transaction_hour": list(np.arange(-1, 24)), "amt_vs_avg": [0, .25, .5, 1, 2, 3, 5, 8, 15, 1e9],
        "t1_share": [-.01, 0, .1, .2, .3, .5, .75, 1.01]}
rows = []
def add(name, key):
    g = pd.DataFrame({"k": key, "y": Y, "p": p}).dropna(subset=["k"]).groupby("k", observed=True)
    for k, s in g:
        if len(s) < 30: continue
        e = s.p.sum(); v = (s.p * (1 - s.p)).sum(); rows.append((name, str(k), len(s), int(s.y.sum()), round(e, 1), (s.y.sum() - e) / np.sqrt(v + 1e-9)))
for c, b in bins.items(): add(c, pd.cut(d[c], b))
for c in ["merchant_category", "country", "transaction_channel", "new_device"]: add(c, d[c])
cats = {"nd": d.new_device.fillna(-1).astype(int).astype(str), "chan": d.transaction_channel.fillna("NA"), "cty": d.country.fillna("NA"),
        "cat": d.merchant_category.fillna("NA"), "night": d.transaction_hour.isin([23, 0, 1, 2, 3, 4]).map({True: "night", False: "day"}),
        "age": pd.cut(d.account_age, [-1, 90, 365, 1500, 1e9]).astype(str), "amtb": pd.cut(d.amt_rel, [0, 1, 3, 1e9]).astype(str),
        "t1": pd.cut(d.transactions_last_1h, [-1, 0, 2, 4, 10]).astype(str), "t24": pd.cut(d.transactions_last_24h, [0, 5, 10, 15, 30]).astype(str)}
for a, b in itertools.combinations(cats, 2): add(f"{a}x{b}", cats[a] + "|" + cats[b])
R = pd.DataFrame(rows, columns=["feature", "bin", "n", "fraud", "expected", "z"])
R.to_csv("experiments/resid_scan.csv", index=False)
pd.set_option("display.width", 200)
print("single-feature bins |z|>2:"); print(R[~R.feature.str.contains("x") & (R.z.abs() > 2)].sort_values("z").to_string(index=False))
print("pairwise cells |z|>3 (", len(R[R.feature.str.contains("x")]), "cells tested):"); print(R[R.feature.str.contains("x") & (R.z.abs() > 3)].sort_values("z").to_string(index=False))
