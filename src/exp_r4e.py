import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap
from sklearn.linear_model import LogisticRegression
from evaluate import Y, W_HN, W_COV, HOLD, TR, SPLITS, SPLITS2, boot_diff
import lrlab, exp_r4d_models as M
v4 = np.load("experiments/oof5_v4.npy"); cb = np.load("experiments/oof5_catboost_d4.npy"); lg = np.load("experiments/oof5_lgb_mono.npy")
# holdout predictions (train on 80% least test-like)
a, b = TR[~HOLD], TR[HOLD]; yh = Y[HOLD]; wh = W_HN[HOLD]
H = {"v4": M.v4(a, b), "cb": M.cb(a, b), "lg": M.lg(a, b)}
def views(o, hp):
    return dict(std=np.mean([ap(Y, o[r]) for r in range(5)]), hnW=np.mean([ap(Y, o[r], sample_weight=W_HN) for r in range(5)]),
                covW=np.mean([ap(Y, o[r], sample_weight=W_COV) for r in range(5)]), hold=ap(yh, hp), hold_hn=ap(yh, hp, sample_weight=wh))
rows = [("v4", *views(v4, H["v4"]).values())]
def rk(x): return np.vstack([rankdata(r) / len(r) for r in x]) if x.ndim == 2 else rankdata(x) / len(x)
for nm, other in [("cb", cb), ("lg", lg)]:
    for w in [0.95, 0.9, 0.8, 0.7, 0.6]:
        rows.append((f"prob {w} v4 + {1-w:.2f} {nm}", *views(w * v4 + (1 - w) * other, w * H["v4"] + (1 - w) * H[nm]).values()))
        rows.append((f"rank {w} v4 + {1-w:.2f} {nm}", *views(w * rk(v4) + (1 - w) * rk(other), w * rk(H["v4"]) + (1 - w) * rk(H[nm])).values()))
# logit-average (geometric) blend
lgt = lambda p: np.log(np.clip(p, 1e-6, 1) / np.clip(1 - p, 1e-6, 1))
for w in [0.8, 0.7]:
    rows.append((f"logit {w} v4 + {1-w:.2f} lg", *views(w * lgt(v4) + (1 - w) * lgt(lg), w * lgt(H["v4"]) + (1 - w) * lgt(H["lg"])).values()))
df = pd.DataFrame(rows, columns=["blend", "std", "hnW", "covW", "holdout", "holdout_hnW"]); print(df.round(4).to_string(index=False))
# stacking: LR meta on logits of OOF, evaluated by CV over rows (meta trained on other folds' OOF)
from sklearn.model_selection import StratifiedKFold
st = []
for r in range(5):
    X = np.c_[lgt(v4[r]), lgt(cb[r]), lgt(lg[r])]; o = np.zeros(len(Y))
    for i, j in StratifiedKFold(5, shuffle=True, random_state=r).split(X, Y):
        o[j] = LogisticRegression(C=1.0).fit(X[i], Y[i]).predict_proba(X[j])[:, 1]
    st.append(o)
st = np.array(st); print("stack LR(v4,cb,lg): std %.4f hnW %.4f" % (np.mean([ap(Y, s) for s in st]), np.mean([ap(Y, s, sample_weight=W_HN) for s in st])))
print("meta coefs (full fit, rep0):", LogisticRegression(C=1.0).fit(np.c_[lgt(v4[0]), lgt(cb[0]), lgt(lg[0])], Y).coef_.round(3))
# disagreement analysis v4 vs lgb (avg OOF)
p1, p2 = v4.mean(0), lg.mean(0); r1, r2 = rankdata(p1) / len(p1), rankdata(p2) / len(p2)
grp = np.select([(r1 > .9) & (r2 > .9), (r1 > .9) & (r2 <= .7), (r1 <= .7) & (r2 > .9), (r1 <= .7) & (r2 <= .7)], ["both_high", "LR_high_tree_low", "LR_low_tree_high", "both_low"], "mixed")
d = TR.assign(g=grp, young=TR.account_age < 90, burst=TR.transactions_last_24h >= 16, big=TR.transaction_amount > 1000)
print(d.groupby("g").agg(n=("fraud", "size"), fraud=("fraud", "sum"), rate=("fraud", "mean"), new_dev=("new_device", "mean"), young=("young", "mean"),
                          burst=("burst", "mean"), big=("big", "mean"), age=("account_age", "median"), amt=("transaction_amount", "median"),
                          bank=("transaction_channel", lambda s: (s == "bank_transfer").mean()), cash=("merchant_category", lambda s: (s == "cash_transfer").mean())).round(3))
for nm, o in [("prob 0.8/0.2 lg", 0.8 * v4 + 0.2 * lg), ("prob 0.7/0.3 lg", 0.7 * v4 + 0.3 * lg)]:
    (lo, m, hi), pw = boot_diff(o.mean(0), v4.mean(0), W_HN); (lo2, m2, hi2), pw2 = boot_diff(o.mean(0), v4.mean(0))
    print(f"{nm}: d_hnW 90%CI {lo:+.4f}..{hi:+.4f} P>0 {pw:.2f} | d_std {lo2:+.4f}..{hi2:+.4f} P>0 {pw2:.2f}")
