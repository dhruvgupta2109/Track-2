import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from features_prod import make_features; from constants import fit_constants
CLEAN = [5, 6, 7, 21, 22]
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv"); K = fit_constants(tr); D = make_features(te, K); Dtr = make_features(tr, K); y = tr.fraud.values
k3 = pd.read_csv("../Track2_K3_final_package_0.223/predictions_K3.csv"); assert (k3.id == te.id).all(); k3 = k3.prediction.values
pinj = np.load("experiments/p_inj_test.npy")
def groups(X):
    b = X.transactions_last_24h >= 16; a = X.account_age; nd = X.new_device == 1; t1 = X.transactions_last_1h >= 3; n = X.transaction_hour.isin([22, 23, 0, 1, 2, 3, 4])
    hv = X.merchant_category.isin(["luxury", "cash_transfer", "electronics", "travel"]) & (X.transaction_amount >= 2000) & (X.transaction_amount <= 6000) & (a > 365)
    return pd.Series(np.select([b & (a <= 365), b & ~(a <= 365), nd & (a <= 90) & t1, nd & (a < 180) & n, nd & (a < 180), nd & (a >= 180) & (a <= 365), nd & (a > 365),
                                hv, t1, (a <= 90), X.foreign == 1],
                               ["burst_young", "burst_old", "nd<=90_fast", "nd<180_night", "nd<180_day", "nd180-365", "nd_old", "highvalue_est", "t1h>=3", "young<=90", "foreign"], "other"), index=X.index)
g = groups(D); gtr = groups(Dtr); clean = D.transaction_hour.isin(CLEAN).values; ctr = Dtr.transaction_hour.isin(CLEAN).values
enr = (g.value_counts() / len(te)) / (gtr.value_counts() / len(tr))
# injected estimate via hour fingerprint per group
inj = {}
for k in g.unique():
    se, st = (g == k).values, (gtr == k).values; c = (se & clean).sum(); o = (se & ~clean).sum(); r = (st & ~ctr).sum() / max((st & ctr).sum(), 1)
    inj[k] = (o - r * c) / max(o + c, 1)
rate = pd.Series(y).groupby(gtr.values).mean()
for kk in [120, 360, 720]:
    top = np.argsort(-k3)[:kk]
    out = pd.concat([g.iloc[top].value_counts().rename("in_top"), g.value_counts().rename("n_test"), enr.rename("enrich"), pd.Series(inj).rename("inj_frac_hour"),
                     pd.Series(pinj).groupby(g.values).mean().rename("mean_P_inj_detector"), rate.rename("train_fraud_rate")], axis=1).fillna(0)
    print(f"--- K3 top {kk} ---"); print(out.sort_values("in_top", ascending=False).round(3).to_string())
