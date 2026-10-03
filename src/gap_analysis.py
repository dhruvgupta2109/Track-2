import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); from common import *
from features_prod import make_features; from constants import fit_constants
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
tr, te = load(); y = tr[LABEL].values
p = pd.read_csv("experiments/oof_v3_lr.csv").oof_v3.values; oof = np.load("experiments/oof_v3_lr.npy")
rng = np.random.default_rng(0)
print("=== A. PR-AUC vs prevalence (subsampling OOF, 200 draws; uses each repeat's OOF) ===")
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; rows = []
for prev in [0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0177, 0.02, 0.025, 0.03, 0.04]:
    aps, aucs = [], []
    for k in range(200):
        o = oof[k % 5]
        if prev <= y.mean():
            npos = int(round(prev * len(neg) / (1 - prev))); idx = np.r_[rng.choice(pos, npos, replace=False), neg]
        else:
            nneg = int(round(len(pos) * (1 - prev) / prev)); idx = np.r_[pos, rng.choice(neg, nneg, replace=False)]
        aps.append(ap(y[idx], o[idx])); aucs.append(auc(y[idx], o[idx]))
    rows.append((prev, np.mean(aps), np.percentile(aps, 5), np.percentile(aps, 95), np.mean(aucs)))
pv = pd.DataFrame(rows, columns=["prevalence", "pr_auc_mean", "p5", "p95", "roc_auc"]); print(pv.round(4).to_string(index=False)); pv.to_csv("experiments/prevalence_sensitivity.csv", index=False)
# bootstrap noise of a 12000-row public set at train prevalence
bs = []
for k in range(300):
    idx = rng.choice(len(y), 12000, replace=True); bs.append(ap(y[idx], oof[k % 5][idx]))
print("AP sampling noise for a 12000-row set at 1.77%%: mean %.4f sd %.4f; 6000 rows: sd ~%.4f" % (np.mean(bs), np.std(bs), np.std(bs) * np.sqrt(2)))

print("\n=== B. adversarial validation (several classifiers, 5-fold OOF) ===")
A = pd.concat([tr.drop(columns=["id", LABEL]), te.drop(columns="id")]).reset_index(drop=True); z = np.r_[np.zeros(len(tr)), np.ones(len(te))]
K = fit_constants(tr); F = make_features(A, K)
num = ["log_amt", "transaction_hour", "transactions_last_24h", "transactions_last_1h", "log_spend", "log_age", "new_device", "amt_rel_cat", "amt_rel_avg"]
miss = [c + "_na" for c in RAW]
for c in RAW: F[c + "_na"] = A[c].isna().astype(int)
Fg = F[num + CATS + miss].copy()
for c in CATS: Fg[c] = Fg[c].astype("category")
skf = StratifiedKFold(5, shuffle=True, random_state=0)
def oof_adv(make, X):
    o = np.zeros(len(X))
    for i, j in skf.split(X, z): o[j] = make().fit(X.iloc[i], z[i]).predict_proba(X.iloc[j])[:, 1]
    return o
res = {}
res["lgb_all"] = oof_adv(lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, verbose=-1), Fg)
res["lgb_no_missing"] = oof_adv(lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, verbose=-1), Fg[num + CATS])
res["lgb_missing_only"] = oof_adv(lambda: lgb.LGBMClassifier(n_estimators=100, verbose=-1), Fg[miss])
ct = ColumnTransformer([("c", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(handle_unknown="ignore")), CATS),
                        ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num + miss)])
res["logreg"] = oof_adv(lambda: make_pipeline(ct, LogisticRegression(C=1, max_iter=3000)), F[num + CATS + miss])
for k, v in res.items(): print(f"adv ROC-AUC {k:18s} {auc(z, v):.4f}")
for c in num + CATS:
    m = lgb.LGBMClassifier(n_estimators=150, verbose=-1)
    X1 = Fg[[c]]; o = np.zeros(len(z))
    for i, j in skf.split(X1, z): o[j] = m.fit(X1.iloc[i], z[i]).predict_proba(X1.iloc[j])[:, 1]
    print(f"   single-feature adv AUC {c:22s} {auc(z, o):.4f}")
mm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, verbose=-1).fit(Fg[num + CATS], z)
print("gain importance (no-missing model):", sorted(zip(mm.booster_.feature_importance("gain").round(0), num + CATS), reverse=True))

print("\n=== C. missingness ===")
mt = pd.DataFrame({"train": tr[RAW].isna().mean(), "test": te[RAW].isna().mean()}); mt["diff"] = mt.test - mt.train; mt["ratio"] = mt.test / mt.train.replace(0, np.nan); print(mt.round(4))
for d, n in [(tr, "train"), (te, "test")]:
    k = d[RAW].isna().sum(1); print(n, "n_missing dist", k.value_counts(normalize=True).sort_index().round(4).to_dict())
print("fraud rate by n_missing (train)", tr.assign(k=tr[RAW].isna().sum(1)).groupby("k").fraud.agg(["size", "mean"]).round(4).to_dict())
print("missing in test by segment: n_missing>0 rate | burst", te[te.transactions_last_24h >= 16][RAW].isna().any(axis=1).mean().round(3),
      "| new_device", te[te.new_device == 1][RAW].isna().any(axis=1).mean().round(3), "| all", te[RAW].isna().any(axis=1).mean().round(3))
print("missingness co-occurrence in test (pairs share):"); M = te[RAW].isna().astype(int); co = M.T @ M; print(co.loc[co.sum(1) > 0, co.sum(0) > 0])

print("\n=== D. fraud-model performance vs test-likeness (train rows, adv OOF lgb_no_missing) ===")
tl = res["lgb_no_missing"][:len(tr)]; np.save("experiments/testlikeness_train.npy", tl); np.save("experiments/testlikeness_test.npy", res["lgb_no_missing"][len(tr):])
q = pd.qcut(pd.Series(tl).rank(method="first"), [0, .2, .4, .6, .8, .9, .95, 1.0], labels=["0-20", "20-40", "40-60", "60-80", "80-90", "90-95", "95-100"])
out = []
for g in q.cat.categories:
    m = (q == g).values
    out.append((g, m.sum(), y[m].sum(), y[m].mean(), np.mean([ap(y[m], oof[r][m]) for r in range(5)]), auc(y[m], p[m])))
print(pd.DataFrame(out, columns=["testlike_pct", "n", "fraud", "rate", "AP", "ROC"]).round(4).to_string(index=False))
# what our model does on test: composition of top-ranked test rows
s = pd.read_csv("submissions/sub_v3_lr.csv"); T = te.assign(p=s.prediction.values)
T["burst"] = T.transactions_last_24h >= 16; T["young"] = T.account_age < 90; T["anymiss"] = te[RAW].isna().any(axis=1)
Tr = tr.assign(p=p, burst=tr.transactions_last_24h >= 16, young=tr.account_age < 90)
for frac in [0.01, 0.02, 0.05]:
    a = T.nlargest(int(frac * len(T)), "p"); b = Tr.nlargest(int(frac * len(Tr)), "p")
    print(f"top {frac:.0%}: TEST burst {a.burst.mean():.2f} new_dev {a.new_device.mean():.2f} young {a.young.mean():.2f} | TRAIN-OOF burst {b.burst.mean():.2f} new_dev {b.new_device.mean():.2f} young {b.young.mean():.2f} (train precision {b.fraud.mean():.2f})")
print("predicted-prob threshold of train-top-2% =", np.quantile(p, .98).round(3), "; share of TEST above it:", (T.p > np.quantile(p, .98)).mean().round(4))
