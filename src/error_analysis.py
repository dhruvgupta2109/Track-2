import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); from common import *
from sklearn.metrics import precision_recall_curve, average_precision_score as ap, roc_auc_score, brier_score_loss, log_loss
pd.set_option("display.width", 200)
tr, te = load(); y = tr[LABEL].values
oof = np.load("experiments/oof_v3_lr.npy"); p = oof.mean(0)   # repeat-averaged OOF
print("AP", ap(y, p), "ROC", roc_auc_score(y, p), "brier", brier_score_loss(y, p), "logloss", log_loss(y, p), "prior-only brier", brier_score_loss(y, np.full(len(y), y.mean())))
# calibration
q = pd.qcut(p, 10, labels=False); cal = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
print("calibration by decile\n", cal.round(4).T)
top = pd.DataFrame({"p": p, "y": y}).sort_values("p", ascending=False)
print("top-bin calibration: top 1% pred", top.p[:200].mean().round(3), "obs", top.y[:200].mean().round(3))
# operating points
pr, rc, th = precision_recall_curve(y, p); f1 = 2 * pr * rc / (pr + rc + 1e-12); i = np.nanargmax(f1[:-1])
print(f"best F1 {f1[i]:.3f} at thr {th[i]:.3f}: precision {pr[i]:.3f} recall {rc[i]:.3f}, flagged {(p>=th[i]).mean()*100:.2f}%")
for k in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
    n = int(k * len(y)); t = top.iloc[:n]
    print(f"flag top {k*100:4.1f}% ({n:5d}): precision {t.y.mean():.3f} recall {t.y.sum()/y.sum():.3f}  lift {t.y.mean()/y.mean():.1f}x  FP per TP {(n-t.y.sum())/max(t.y.sum(),1):.1f}")
for P in [0.5, 0.3, 0.2]:
    ok = pr[:-1] >= P; print(f"max recall at precision>={P}: {rc[:-1][ok].max() if ok.any() else 0:.3f}")
# segment analysis at the 5% flag rate
thr5 = top.p.iloc[int(0.05 * len(y)) - 1]
d = tr.copy(); d["p"] = p; d["flag"] = p >= thr5
d["age_bin"] = pd.cut(d.account_age, [0, 90, 365, 1500, 5000]).astype(str); d["burst"] = d.transactions_last_24h >= 16
d["amt_bin"] = pd.qcut(d.transaction_amount, 4).astype(str)
def seg(c):
    g = d.groupby(c, dropna=False).apply(lambda s: pd.Series({"n": len(s), "fraud": s.fraud.sum(), "rate": s.fraud.mean(), "recall@5%": s[s.fraud == 1].flag.mean() if s.fraud.sum() else np.nan,
        "precision@5%": s[s.flag].fraud.mean() if s.flag.any() else np.nan, "flag_rate": s.flag.mean(), "AP": ap(s.fraud, s.p) if 0 < s.fraud.sum() < len(s) else np.nan}))
    print(g.round(3))
for c in ["transaction_channel", "merchant_category", "new_device", "age_bin", "burst", "amt_bin", "country"]: seg(c)
# false negatives: frauds ranked below median
d["rank_pct"] = pd.Series(p).rank(pct=True).values
fn = d[(d.fraud == 1)]
print("fraud rank percentile quantiles", fn.rank_pct.quantile([.1, .25, .5, .75]).round(3).to_dict())
print("share of frauds ranked below 50th pct:", (fn.rank_pct < .5).mean().round(3))
hard = fn[fn.rank_pct < .5]; easy = fn[fn.rank_pct >= .9]
cols = ["transaction_amount", "account_age", "transactions_last_24h", "transactions_last_1h", "new_device"]
print("hard FN profile (median)\n", hard[cols].median(), "\nlegit median\n", d[d.fraud == 0][cols].median())
print("hard FN channel/category", hard.transaction_channel.value_counts().to_dict(), hard.merchant_category.value_counts().head(5).to_dict())
fp = d[(d.fraud == 0)].sort_values("p", ascending=False).head(100)
print("top-100 high-confidence FPs: new_device", fp.new_device.mean(), "burst", fp.burst.mean(), "young", (fp.account_age < 90).mean(), "median amt", fp.transaction_amount.median())
pd.DataFrame({"id": tr.id, "fraud": y, "oof_v3": p}).to_csv("experiments/oof_v3_lr.csv", index=False)
