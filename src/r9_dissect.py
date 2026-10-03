import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from evaluate import TR, SPLITS, Y
import r6lab
from features_prod import make_features; from constants import fit_constants
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
EMB = r6lab.EMB; o5 = np.load("experiments/oof5_v5_prod.npy")[0]; o5a = np.load("experiments/oof5_v5a_prod.npy")[0]
rows = []
for a_idx, b_idx in SPLITS[:5]:
    a, b = TR.iloc[a_idx], TR.iloc[b_idx]; K = fit_constants(a); A = make_features(a, K); B = make_features(b, K); ya = a.fraud.values
    med = A[EMB].median(); mu = A[EMB].fillna(med).mean(); sd = A[EMB].fillna(med).std(ddof=0)
    Ea = ((A[EMB].fillna(med) - mu) / sd).values; Eb = ((B[EMB].fillna(med) - mu) / sd).values
    nf = NearestNeighbors(n_neighbors=1).fit(Ea[ya == 1]); nl = NearestNeighbors(n_neighbors=1).fit(Ea[ya == 0])
    df_, jf = nf.kneighbors(Eb); dl_, jl = nl.kneighbors(Eb)
    Ff = Ea[ya == 1][jf[:, 0]]; Fl = Ea[ya == 0][jl[:, 0]]
    diff_f = (Eb - Ff) ** 2; diff_l = (Eb - Fl) ** 2
    raw_f = a[ya == 1].iloc[jf[:, 0]]
    for k, i in enumerate(b_idx):
        r = {"i": i, "y": Y[i], "d_f": df_[k, 0], "d_l": dl_[k, 0], "p5": o5[i], "p5a": o5a[i],
             "same_chan_f": b.transaction_channel.iloc[k] == raw_f.transaction_channel.iloc[k], "same_cat_f": b.merchant_category.iloc[k] == raw_f.merchant_category.iloc[k],
             "same_cty_f": b.country.iloc[k] == raw_f.country.iloc[k], "same_hour_f": b.transaction_hour.iloc[k] == raw_f.transaction_hour.iloc[k]}
        for j, c in enumerate(EMB): r["sqd_f_" + c] = diff_f[k, j]; r["sqd_l_" + c] = diff_l[k, j]
        rows.append(r)
D = pd.DataFrame(rows); D["ratio"] = np.log1p(D.d_f) - np.log1p(D.d_l)
D["lift"] = np.log(D.p5 / (1 - D.p5)) - np.log(D.p5a / (1 - D.p5a))   # how much kNN moved the row (logit)
print("nearest-fraud distance quantiles: fraud rows", np.round(D[D.y == 1].d_f.quantile([.1, .25, .5, .75]).values, 3), "legit rows", np.round(D[D.y == 0].d_f.quantile([.1, .25, .5, .75]).values, 3))
print("nearest-legit distance quantiles: fraud rows", np.round(D[D.y == 1].d_l.quantile([.1, .25, .5, .75]).values, 3), "legit rows", np.round(D[D.y == 0].d_l.quantile([.1, .25, .5, .75]).values, 3))
print("fraud rate by d_f decile:", D.groupby(pd.qcut(D.d_f, 10, labels=False)).y.mean().round(4).tolist())
print("fraud rate by ratio decile:", D.groupby(pd.qcut(D.ratio, 10, labels=False)).y.mean().round(4).tolist())
# very close fraud neighbours (near-duplicates?)
for t in [0.05, 0.1, 0.2, 0.3]:
    m = D.d_f < t; print(f"d_f<{t}: n={m.sum()} fraud rate {D[m].y.mean():.3f}")
q = D.lift.quantile([.02, .98])
grp = {"fraud promoted": (D.y == 1) & (D.lift > q[.98] * 0 + 0.3), "fraud demoted": (D.y == 1) & (D.lift < -0.3),
       "legit demoted": (D.y == 0) & (D.lift < -0.3), "legit promoted": (D.y == 0) & (D.lift > 0.3)}
cols = [c for c in D.columns if c.startswith("sqd_f_")]
print("\nshare of squared nearest-FRAUD distance by dimension (mean over group):")
T = {}
for g, m in grp.items():
    s = D.loc[m, cols]; T[g] = (s.div(s.sum(1), axis=0)).mean().values
    print(f"  {g:15s} n={m.sum():5d}  same channel {D[m].same_chan_f.mean():.2f} cat {D[m].same_cat_f.mean():.2f} country {D[m].same_cty_f.mean():.2f} hour {D[m].same_hour_f.mean():.2f}  median d_f {D[m].d_f.median():.3f} d_l {D[m].d_l.median():.3f}")
print(pd.DataFrame(T, index=[c[6:] for c in cols]).round(3))
print("\nbaseline same-X rates for all rows: channel %.2f cat %.2f country %.2f" % (D.same_chan_f.mean(), D.same_cat_f.mean(), D.same_cty_f.mean()))
print("\nfraud rate by (same channel as nearest fraud, ratio tercile):"); print(D.groupby([D.same_chan_f, pd.qcut(D.ratio, 3, labels=False)]).y.agg(["size", "mean"]).round(4))
print("fraud rate by (same merchant as nearest fraud, ratio tercile):"); print(D.groupby([D.same_cat_f, pd.qcut(D.ratio, 3, labels=False)]).y.agg(["size", "mean"]).round(4))
D.to_csv("experiments/r9_knn_dissect.csv", index=False)
