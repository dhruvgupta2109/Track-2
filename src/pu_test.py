import sys, warnings, numpy as np, pandas as pd, hashlib; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from pu import pu_fit_predict
from features_prod import make_features; from constants import fit_constants
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv"); D = make_features(te, fit_constants(tr))
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
v5 = pd.read_csv("submissions/sub_v5_lr_knn.csv").prediction.values
b = (D.transactions_last_24h >= 16).values; a_ = D.account_age.values; nd = (D.new_device == 1).values
seg = {"burst": b, "nd_old": nd & (a_ > 365) & ~b, "nd_young": nd & (a_ <= 365) & ~b, "big&age>1500": (D.amt_rel_cat > 1).values & (a_ > 1500) & ~b & ~(nd & (a_ > 365)),
       "travel": (D.merchant_category == "travel").values, "t24 10-15": ((D.transactions_last_24h >= 10) & (D.transactions_last_24h < 16)).values,
       "foreign": (D.foreign == 1).values, "night": (D.night == 1).values, "luxury": (D.merchant_category == "luxury").values, "young<90": a_ < 90}
known = {"burst": "DOWN helped (-1.0)", "nd_old": "DOWN helped (-0.75)", "big&age>1500": "down HURT", "travel": "down hurt slightly"}
for alpha, nm in [(0.0, "PU_a0"), (0.1, "PU_a0.1")]:
    p = pu_fit_predict(tr, te, alpha=alpha, n_mask=3, rate=0.017)
    d = lg(p) - lg(v5); d = d - np.median(d)     # PU odds have an arbitrary offset; only relative shifts matter for ranking
    print(f"--- {nm}: implied per-segment shift (mean logit PU - v5, centred) ---")
    for k, m in seg.items(): print(f"   {k:14s} n={m.sum():5d}  {d[m].mean():+.2f}   {known.get(k, '')}")
    f = f"submissions/shift/{nm}.csv"; pd.DataFrame({"id": te.id, "prediction": p}).to_csv(f, index=False)
    print("   hash", hashlib.sha256(open(f, "rb").read()).hexdigest()[:10])
