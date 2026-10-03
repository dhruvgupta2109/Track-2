"""Legit-drift density-ratio correction (class-conditional shift: P(x|fraud) fixed, P(x|legit) drifts in the test base).
log-odds_test(x) = log-odds_train(x) - log r(x),  r(x) = p_testbase(x | legit) / p_train(x | legit) ~ p_testbase(x) / p_train(x | legit).
r is estimated by a classifier train-legit vs test-base rows (test rows weighted by 1 - P(injected); hour excluded), cross-fitted on test."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
sys.path.insert(0, "src")
import models
from features_prod import make_features, CAT_COLS
from constants import fit_constants

tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv"); K = fit_constants(tr)
A = make_features(tr, K); B = make_features(te, K)
for D in (A, B):
    D["age_le365"] = (D.account_age <= 365).astype(float).where(D.account_age.notna())
    D["burst_young"] = D.burst * D.age_le365; D["burst_logage"] = D.burst * D.log_age
    D["t24_eq1"] = (D.transactions_last_24h == 1).astype(float); D["spend0"] = (D.spend_last_24h == 0).astype(float)
NUM = ["log_amt", "log_spend", "log_age", "new_device", "transactions_last_24h", "transactions_last_1h", "amt_rel_cat", "avg_spend", "amt_rel_avg",
       "young", "burst", "t1_share", "foreign", "age_le365", "burst_young", "burst_logage", "nd_x_logage", "t24_eq1", "spend0"]
P_INJ = np.load("experiments/p_inj_test.npy")
CLEAN = B.transaction_hour.isin([5, 6, 7, 21, 22]).values


def log_ratio(test_w, C=0.3, num=NUM, folds=5, seed=0):
    """cross-fitted log r(x) for every test row; test_w = weight of each test row as 'base' (0 drops it)."""
    L = A[tr.fraud.values == 0]; out = np.zeros(len(B))
    for k, (a, b) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(B)):
        keep = a[test_w[a] > 0]
        X = pd.concat([L, B.iloc[keep]]); z = np.r_[np.zeros(len(L)), np.ones(len(keep))]; w = np.r_[np.ones(len(L)), test_w[keep]]
        m = models.lr_pipeline(C, num, CAT_COLS).fit(X[CAT_COLS + num], z, logisticregression__sample_weight=w)
        pr = np.clip(m.predict_proba(B.iloc[b][CAT_COLS + num])[:, 1], 1e-6, 1 - 1e-6)
        out[b] = np.log(pr / (1 - pr)) - np.log(w[len(L):].sum() / len(L))
    return out


def groups(D):
    b = (D.transactions_last_24h >= 16).values; a_ = D.account_age.values; nd = (D.new_device == 1).values; t1 = (D.transactions_last_1h >= 3).values
    ndold = nd & (a_ > 365) & ~b
    hv = D.merchant_category.isin(["luxury", "cash_transfer", "electronics", "travel"]).values & (D.transaction_amount >= 2000).values & (D.transaction_amount <= 6000).values & (a_ > 365) & ~b & ~ndold
    return {"burst_young [-2.0]": b & (a_ <= 365), "burst_old [-0.5]": b & ~(a_ <= 365), "nd_old [-0.75]": ndold, "highvalue_est [+0.5]": hv,
            "travel [~0]": (D.merchant_category == "travel").values & ~b & ~ndold & ~hv, "nd<=90_fast [~0]": nd & (a_ <= 90) & t1 & ~b,
            "foreign [?]": (D.foreign == 1).values & ~b & ~ndold & ~hv, "amt>2000 [?]": (D.transaction_amount > 2000).values & ~hv & ~b}


if __name__ == "__main__":
    G = groups(B)
    lv5 = np.log(np.clip(pd.read_csv("submissions/sub_v5_lr_knn.csv").prediction.values, 1e-9, 1))
    for name, w in [("base=1-p_inj", 1 - P_INJ), ("base=clean hours only", CLEAN.astype(float))]:
        for C in (0.1, 1.0):
            lr = log_ratio(w, C); c = -(lr - np.median(lr))
            np.save(f"experiments/legit_drift_{'clean' if 'clean' in name else 'pinj'}_C{C}.npy", lr)
            print(f"{name} C={C}: correction sd {c.std():.2f}; group means (centred): " + "  ".join(f"{k} {c[m].mean():+.2f}" for k, m in G.items()), flush=True)
            # where does the correction act among the top ranks of v5?
            top = np.argsort(-lv5)[:300]; print(f"      top-300 v5 rows: mean corr {c[top].mean():+.2f}, sd {c[top].std():.2f}")
