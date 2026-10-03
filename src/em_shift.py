"""Saerens-Latinne-Decaestecker (2002) EM for label (prior) shift, applied per segment of the test set."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from evaluate import TR, TE
from features_prod import make_features; from constants import fit_constants
lg = lambda p: np.log(p / (1 - p)); sg = lambda z: 1 / (1 + np.exp(-z))
def em_prior(p, pi_tr, iters=200):
    pi = pi_tr
    for _ in range(iters):
        q = sg(lg(np.clip(p, 1e-9, 1 - 1e-9)) + lg(pi) - lg(pi_tr)); pi_new = q.mean()
        if abs(pi_new - pi) < 1e-7: break
        pi = pi_new
    return pi
def segments(D):
    a = D.account_age; big = D.amt_rel_cat > 1.0; nd = D.new_device == 1; b = D.burst == 1
    return {"ALL": np.ones(len(D), bool), "burst": b, "non-burst": ~b, "new_device": nd, "nd & age>365": nd & (a > 365), "nd & age<=365": nd & (a <= 365),
            "no nd": ~nd & ~b, "big & age>1500": big & (a > 1500), "big & age<=1500": big & (a <= 1500), "t1h>=5 non-burst": (D.transactions_last_1h >= 5) & ~b,
            "young<90": a < 90, "age>1500": a > 1500, "foreign": D.foreign == 1, "SG": D.foreign == 0, "night": D.night == 1,
            "any_missing": D[["merchant_category", "country", "transaction_channel", "transactions_last_24h", "spend_last_24h", "account_age", "new_device", "transactions_last_1h"]].isna().any(axis=1).values,
            **{f"cat={c}": (D.merchant_category == c).values for c in sorted(D.merchant_category.dropna().unique())},
            **{f"chan={c}": (D.transaction_channel == c).values for c in sorted(D.transaction_channel.dropna().unique())}}
if __name__ == "__main__":
    K = fit_constants(TR); Dtr = make_features(TR, K); Dte = make_features(TE, K)
    oof = pd.read_csv("experiments/oof_v3_v4_ranks.csv")  # contains v4 OOF; use v5 OOF from npy instead
    p_tr = np.load("experiments/oof5_v5_prod.npy").mean(0); p_te = pd.read_csv("submissions/sub_v5_lr_knn.csv").prediction.values
    Str, Ste = segments(Dtr), segments(Dte); rows = []
    for k in Str:
        mtr, mte = np.asarray(Str[k], bool), np.asarray(Ste[k], bool)
        if mte.sum() < 30 or mtr.sum() < 30: continue
        rate_tr = TR.fraud.values[mtr].mean(); calib = p_tr[mtr].mean()      # train: observed vs OOF-predicted rate in segment
        pi_te = em_prior(p_te[mte], max(calib, 1e-4))
        rows.append((k, mtr.sum(), mte.sum(), round(rate_tr, 4), round(calib, 4), round(p_te[mte].mean(), 4), round(pi_te, 4), round(lg(max(pi_te, 1e-5)) - lg(max(calib, 1e-5)), 2)))
    df = pd.DataFrame(rows, columns=["segment", "n_train", "n_test", "train_rate", "oof_mean_p", "test_mean_p", "EM_test_prior", "implied_logit_shift"])
    pd.set_option("display.width", 200); print(df.to_string(index=False)); df.to_csv("experiments/em_shift_segments.csv", index=False)
