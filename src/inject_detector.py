"""Per-row P(injected | x) for test rows, learned label-free from the hour fingerprint (clean hours 5,6,7,21,22 = pure base).
For each band (night 23-4, day 8-20): cross-fitted LightGBM 'band vs clean' on non-hour features.
odds(x) = [B_b p(x) + E_b q(x)] / [B_c p(x)]  =>  P(inj|x) = 1 - (B_b/B_c) / odds(x)."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "src")
from features_prod import make_features; from constants import fit_constants
CLEAN, NIGHT, DAY = [5, 6, 7, 21, 22], [23, 0, 1, 2, 3, 4], list(range(8, 21))
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv"); D = make_features(te, fit_constants(tr))
FEATS = ["transaction_amount", "log_amt", "amt_rel_cat", "account_age", "new_device", "transactions_last_24h", "transactions_last_1h",
         "spend_last_24h", "amt_rel_avg", "merchant_category", "country", "transaction_channel"]
X = D[FEATS].copy()
for c in ["merchant_category", "country", "transaction_channel"]: X[c] = X[c].astype("category")
h = D.transaction_hour.values; clean = np.isin(h, CLEAN)
p_inj = np.zeros(len(te))
for band, nh in [(NIGHT, 6), (DAY, 13)]:
    m = np.isin(h, band); idx = np.where(m | clean)[0]; z = m[idx].astype(int)
    odds = np.zeros(len(idx))
    for seed in range(3):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(idx, z):
            g = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=seed).fit(X.iloc[idx[a]], z[a])
            pr = np.clip(g.predict_proba(X.iloc[idx[b]])[:, 1], 1e-4, 1 - 1e-4); odds[b] += pr / (1 - pr) / 3
    ratio = nh / 5.0
    pi = np.clip(1 - ratio / odds, 0, 0.98)
    p_inj[idx[z == 1]] = pi[z == 1]
    print(f"band {band[0]}..{band[-1]}: rows {m.sum()}, mean P(inj) {pi[z == 1].mean():.3f} (fingerprint estimate {'0.25' if nh == 6 else '0.29'}), "
          f"share with P(inj)>0.5: {(pi[z == 1] > 0.5).mean():.2f}")
np.save("experiments/p_inj_test.npy", p_inj)
b = (D.transactions_last_24h >= 16).values; a_ = D.account_age.values; nd = (D.new_device == 1).values
seg = {"burst": b, "nd_old": nd & (a_ > 365) & ~b, "big&age>1500": (D.amt_rel_cat > 1).values & (a_ > 1500) & ~b & ~(nd & (a_ > 365)),
       "travel": (D.merchant_category == "travel").values, "young<=90": a_ <= 90, "clean hours": clean}
print("mean log(1-P(inj)) by segment (implied shift if injected rows are legit):")
for k, s in seg.items(): print(f"   {k:14s} n={s.sum():5d}  {np.log(1 - p_inj[s]).mean():+.2f}")
# within big&age>1500: how does P(inj) relate to v5 score? (do high-scoring big/old rows look injected?)
v5 = pd.read_csv("submissions/sub_v5_lr_knn.csv").prediction.values; s = seg["big&age>1500"]
q = pd.qcut(v5[s], 4, labels=["low", "mid", "high", "top"]); print("big&age>1500: mean P(inj) by v5-score quartile:", pd.Series(p_inj[s]).groupby(q.astype(str)).mean().round(2).to_dict())
