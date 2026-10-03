"""MLLS/EM label-shift estimation (Saerens 2002) with bias-corrected calibration (Alexandari et al. 2020), per test sub-population.
Calibration: Platt/BCTS (a*z+b) fitted on train OOF logits of v5. For each group g, EM estimates the test prior from
calibrated posteriors whose training prior is the group's train fraud rate. Implied shift = logit(pi_test) - logit(pi_train)."""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "src")
from features_prod import make_features; from constants import fit_constants
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv"); K = fit_constants(tr); Dtr, Dte = make_features(tr, K), make_features(te, K); y = tr.fraud.values
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1)); sg = lambda z: 1 / (1 + np.exp(-z))
oof = np.load("experiments/oof5_v5_prod.npy").mean(0); v5 = pd.read_csv("submissions/sub_v5_lr_knn.csv").prediction.values
cal = LogisticRegression(C=1e6).fit(lg(oof)[:, None], y); a, b = cal.coef_[0, 0], cal.intercept_[0]
print(f"BCTS/Platt calibration on train OOF: a={a:.3f} b={b:.3f}")
ptr, pte = sg(a * lg(oof) + b), sg(a * lg(v5) + b)
def em(p, pi0, iters=500):
    pi = pi0
    for _ in range(iters):
        q = sg(lg(p) + lg(pi) - lg(pi0)); new = q.mean()
        if abs(new - pi) < 1e-8: break
        pi = new
    return pi
def groups(X):
    bst = (X.transactions_last_24h >= 16).values; a_ = X.account_age.values; nd = (X.new_device == 1).values; t1 = (X.transactions_last_1h >= 3).values
    night = X.transaction_hour.isin([22, 23, 0, 1, 2, 3, 4]).values
    ndold = nd & (a_ > 365) & ~bst
    hv = X.merchant_category.isin(["luxury", "cash_transfer", "electronics", "travel"]).values & (X.transaction_amount >= 2000).values & (X.transaction_amount <= 6000).values & (a_ > 365) & ~bst & ~ndold
    g = {"burst_young": bst & (a_ <= 365), "burst_old": bst & ~(a_ <= 365), "nd_old": ndold, "highvalue_est(K3)": hv,
         "bigold(amt_rel>1,age>1500)": (X.amt_rel_cat > 1).values & (a_ > 1500) & ~bst & ~ndold,
         "travel": (X.merchant_category == "travel").values & ~bst & ~ndold & ~hv, "nd<=90_fast": nd & (a_ <= 90) & t1 & ~bst,
         "nd<180_night_lowvel(K3)": nd & (a_ < 180) & night & ~t1 & (X.transactions_last_24h < 12).values & ~bst,
         "nd<180_day": nd & (a_ < 180) & ~night & ~t1 & ~bst, "nd180-365": nd & (a_ >= 180) & (a_ <= 365) & ~bst,
         "t1h>=3 (no nd)": t1 & ~nd & ~bst, "young<=90 (no nd)": (a_ <= 90) & ~nd & ~t1 & ~bst, "foreign other": (X.foreign == 1).values & ~nd & ~bst & ~t1 & ~(a_ <= 90),
         "night other": night & ~nd & ~bst & ~t1 & ~(a_ <= 90), "rest": None}
    used = np.zeros(len(X), bool)
    for k in list(g)[:-1]: g[k] = g[k] & ~used; used |= g[k]
    g["rest"] = ~used
    return g
Gtr, Gte = groups(Dtr), groups(Dte)
public_best = {"burst_young": "-2.0 (0,-1 worse; -3 worse)", "burst_old": "-0.5 (-1 worse, -0.25 ~)", "nd_old": "-0.75 (-0.5,-1.0 slightly worse)",
               "highvalue_est(K3)": "+0.5 (K3)", "bigold(amt_rel>1,age>1500)": "+0.5 (0,-0.5,-1.1 worse)", "travel": "~0 (+-0.5 worse)",
               "nd<=90_fast": "~0 (-1, +0.5, +1 worse)", "nd<180_night_lowvel(K3)": "-0.5 (K3)"}
rows = []
for k in Gtr:
    mtr, mte = Gtr[k], Gte[k]
    pi_tr = y[mtr].mean(); cal_tr = ptr[mtr].mean()            # observed vs calibrated-predicted train rate (sanity)
    pi_te = em(pte[mte], max(cal_tr, 1e-5))
    rows.append((k, mtr.sum(), mte.sum(), round(pi_tr, 4), round(cal_tr, 4), round(pte[mte].mean(), 4), round(pi_te, 4), round(lg(max(pi_te, 1e-5)) - lg(max(cal_tr, 1e-5)), 2), public_best.get(k, "")))
df = pd.DataFrame(rows, columns=["group", "n_tr", "n_te", "train_rate", "train_cal_mean", "test_mean_p", "EM_test_prior", "EM_shift", "public_best"])
df["EM_shift_rel_rest"] = (df.EM_shift - df.loc[df.group == "rest", "EM_shift"].values[0]).round(2)
pd.set_option("display.width", 250); print(df.to_string(index=False))
