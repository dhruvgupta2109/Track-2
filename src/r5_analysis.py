import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from evaluate import Y, TR
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
o3 = np.load("experiments/oof5_v3_lr.npy").mean(0); o4 = np.load("experiments/oof5_v4.npy").mean(0)
d = TR.copy(); n = len(d)
d["p3"], d["p4"] = o3, o4; d["r3"] = rankdata(-o3) / n; d["r4"] = rankdata(-o4) / n   # 0 = top
d["dr"] = d.r3 - d.r4   # >0 = moved up under v4
med = d.groupby("merchant_category").transaction_amount.median()
d["amt_rel"] = np.log1p(d.transaction_amount) - np.log1p(d.merchant_category.map(med))
d["young"] = d.account_age < 90; d["burst"] = d.transactions_last_24h >= 16; d["night"] = d.transaction_hour.isin([23, 0, 1, 2, 3])
d["foreign"] = d.country.notna() & (d.country != "SG"); d["big"] = d.amt_rel > 1
d["age_b"] = pd.cut(d.account_age, [0, 30, 90, 365, 1500, 5000]).astype(str)
def prof(x):
    return pd.Series({"n": len(x), "fraud": x.fraud.sum(), "rate": x.fraud.mean(), "new_dev": x.new_device.mean(), "young": x.young.mean(), "night": x.night.mean(),
                      "burst": x.burst.mean(), "foreign": x.foreign.mean(), "big": x.big.mean(), "age_med": x.account_age.median(), "amt_med": x.transaction_amount.median()})
print("=== 1. v3 -> v4 rank movement ===")
print(pd.DataFrame({"up_top200": prof(d.nlargest(200, "dr")), "down_top200": prof(d.nsmallest(200, "dr")), "all": prof(d)}).round(3))
# segment-level: where did fraud ranks improve? Compare mean rank of frauds and legit per new_device x age x night
g = d[d.new_device == 1].groupby(["age_b", "night"]).apply(lambda x: pd.Series({"n": len(x), "fraud": x.fraud.sum(), "rate": x.fraud.mean(),
        "fraud_rank_v3": x[x.fraud == 1].r3.mean(), "fraud_rank_v4": x[x.fraud == 1].r4.mean(), "legit_rank_v3": x[x.fraud == 0].r3.mean(), "legit_rank_v4": x[x.fraud == 0].r4.mean()}))
print("new_device rows by age x night (rank: 0=top)\n", g.round(3))
# top-of-list composition change: top 2% under each
for k in [0.01, 0.02, 0.05]:
    a = d[d.r3 <= k]; b = d[d.r4 <= k]
    print(f"top {k:.0%}: precision v3 {a.fraud.mean():.3f} v4 {b.fraud.mean():.3f} | entered (n={len(b.index.difference(a.index))}, fraud rate {d.loc[b.index.difference(a.index)].fraud.mean():.3f}) "
          f"left (n={len(a.index.difference(b.index))}, fraud rate {d.loc[a.index.difference(b.index)].fraud.mean():.3f})")
    ent = d.loc[b.index.difference(a.index)]; lef = d.loc[a.index.difference(b.index)]
    print("     entered:", prof(ent)[["new_dev", "young", "night", "burst", "foreign", "big", "age_med"]].round(2).to_dict())
    print("     left:   ", prof(lef)[["new_dev", "young", "night", "burst", "foreign", "big", "age_med"]].round(2).to_dict())

print("\n=== 2. fraud by v4 rank tier ===")
f = d[d.fraud == 1].copy(); f["tier"] = pd.cut(f.r4, [0, 0.05, 0.3, 1.0], labels=["A_top5%", "B_5-30%", "C_below30%"])
print(f.groupby("tier").apply(prof).round(3))
# matched comparison for tiers B and C: legit rows with similar v4 score (same score decile band) and same merchant & channel
print("\nB/C frauds vs score-matched legit (same score band +-2.5% rank, same merchant_category):")
rows = []
for t in ["B_5-30%", "C_below30%"]:
    ff = f[f.tier == t]; L = []
    for _, r in ff.iterrows():
        cand = d[(d.fraud == 0) & ((d.r4 - r.r4).abs() < 0.025) & (d.merchant_category == r.merchant_category)]
        if len(cand): L.append(cand.sample(min(20, len(cand)), random_state=0))
    L = pd.concat(L)
    cols = ["transaction_amount", "amt_rel", "account_age", "transactions_last_24h", "transactions_last_1h", "spend_last_24h", "transaction_hour", "new_device"]
    cmp = pd.DataFrame({"fraud_mean": ff[cols].mean(), "legit_mean": L[cols].mean(), "fraud_med": ff[cols].median(), "legit_med": L[cols].median()})
    print(t, "n_fraud", len(ff), "n_matched_legit", len(L)); print(cmp.round(2))
    for c in ["transaction_channel", "country"]:
        print("  ", c, "fraud:", ff[c].value_counts(normalize=True).round(2).head(5).to_dict(), "| legit:", L[c].value_counts(normalize=True).round(2).head(5).to_dict())
    print("   foreign fraud %.2f legit %.2f | night fraud %.2f legit %.2f | t1h>=3 fraud %.2f legit %.2f" % (ff.foreign.mean(), L.foreign.mean(), ff.night.mean(), L.night.mean(), (ff.transactions_last_1h >= 3).mean(), (L.transactions_last_1h >= 3).mean()))
    print("   spend/tx24 ratio (amt / avg) fraud %.2f legit %.2f" % ((ff.transaction_amount / (ff.spend_last_24h / ff.transactions_last_24h + 1)).median(), (L.transaction_amount / (L.spend_last_24h / L.transactions_last_24h + 1)).median()))

print("\n=== 3. hardest false positives vs frauds in same score range ===")
for k in [0.005, 0.01, 0.02, 0.05]:
    top = d[d.r4 <= k]; fp = top[top.fraud == 0]; tp = top[top.fraud == 1]
    print(f"top {k:.1%}: n={len(top)} precision={top.fraud.mean():.3f}")
    print(pd.DataFrame({"legit": prof(fp), "fraud": prof(tp)}).T[["n", "new_dev", "young", "night", "burst", "foreign", "big", "age_med", "amt_med"]].round(2).to_string())
top = d[d.r4 <= 0.05]
for c in ["transaction_channel", "merchant_category", "age_b"]:
    print(c, "\n", top.groupby(c).fraud.agg(["size", "mean"]).round(3).T.to_string())
print("within top 5%: fraud rate by burst x young:\n", top.groupby(["burst", "young"]).fraud.agg(["size", "mean"]).round(3))
print("within top 5%: fraud rate by burst x night:\n", top.groupby(["burst", "night"]).fraud.agg(["size", "mean"]).round(3))
print("within top 5%: fraud rate by big x age_b:\n", top.groupby(["big", "age_b"]).fraud.agg(["size", "mean"]).round(3))
print("within top 5%: fraud rate by foreign:\n", top.groupby("foreign").fraud.agg(["size", "mean"]).round(3))
print("within top 5%: t1h:\n", top.groupby(top.transactions_last_1h.clip(upper=6)).fraud.agg(["size", "mean"]).round(3).T)
d[["id", "fraud", "p3", "p4", "r3", "r4"]].to_csv("experiments/oof_v3_v4_ranks.csv", index=False)
