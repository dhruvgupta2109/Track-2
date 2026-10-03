"""Adversarial density-ratio weights for train rows -> 'test-like' weighted PR-AUC of saved OOFs."""
import sys, numpy as np, pandas as pd, warnings, glob; warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); from common import *
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as ap, roc_auc_score
tr, te = load(); y = tr[LABEL].values
cols = RAW
a = pd.concat([tr[cols], te[cols]]).reset_index(drop=True); z = np.r_[np.zeros(len(tr)), np.ones(len(te))]
# missingness differences are handled separately (injection); exclude them from the ratio by imputing nothing - LGB uses NaN natively,
# so add the missing pattern? We drop missing-driven signal by filling NaN with a sentinel only for adversarial model:
for c in CATS: a[c] = a[c].astype("category")
p = np.zeros(len(a))
for f, (i, j) in enumerate(StratifiedKFold(5, shuffle=True, random_state=1).split(a, z)):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, verbose=-1).fit(a.iloc[i], z[i])
    p[j] = m.predict_proba(a.iloc[j])[:, 1]
print("adv AUC", roc_auc_score(z, p))
pt = np.clip(p[:len(tr)], 0.02, 0.98); w = pt / (1 - pt) * (len(tr) / len(te)); w = w / w.mean()
print("weight quantiles", np.quantile(w, [0, .1, .5, .9, .99, 1]).round(2), "ESS", (w.sum() ** 2 / (w ** 2).sum()).round(0))
print("weighted fraud prevalence (train reweighted to test covariates):", np.average(y, weights=w).round(4))
np.save("experiments/adv_weights.npy", w)
rows = []
for f in sorted(glob.glob("experiments/oof_*.npy")):
    o = np.load(f); n = f.split("oof_")[1][:-4]
    rows.append((n, np.mean([ap(y, o[r]) for r in range(len(o))]), np.mean([ap(y, o[r], sample_weight=w) for r in range(len(o))])))
lr = np.load("experiments/oof_lr_eng_noind_C0.1.npy"); lg = np.load("experiments/oof_lgb_eng_mono.npy")
for wl in [1.0, 0.8, 0.7, 0.5]:
    b = wl * lr + (1 - wl) * lg
    rows.append((f"blend_lr{wl}", np.mean([ap(y, b[r]) for r in range(5)]), np.mean([ap(y, b[r], sample_weight=w) for r in range(5)])))
print(pd.DataFrame(rows, columns=["model", "pr_auc", "pr_auc_testweighted"]).sort_values("pr_auc_testweighted", ascending=False).round(4).to_string(index=False))
