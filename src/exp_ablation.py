import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); from common import *; import models
from features_prod import make_features; from constants import fit_constants
from sklearn.metrics import average_precision_score as ap
from sklearn.linear_model import LogisticRegression
tr, te = load(); y = tr[LABEL].values; R = 3; splits = cv_splits(y, n_repeats=R)
w = np.load("experiments/adv_weights.npy")
def lr_f(num, cat, C=0.1, cw=None):
    def fp(a, b):
        K = fit_constants(a); A = make_features(a, K); B = make_features(b, K)
        m = models.lr_pipeline(C, num, cat)
        if cw: m.set_params(logisticregression__class_weight=cw)
        m.fit(A[cat + num], a[LABEL]); return m.predict_proba(B[cat + num])[:, 1]
    return fp
N, Cc = models.LR_NUM, models.LR_CAT
res = []
def go(n, fp):
    oof, s = run_cv(fp, tr, splits, R, verbose=False)
    res.append((n, s["pr_auc_mean"], np.mean([ap(y, oof[r], sample_weight=w) for r in range(R)]))); print(res[-1], flush=True)
go("full", lr_f(N, Cc))
go("balanced", lr_f(N, Cc, cw="balanced"))
groups = {"velocity_1h": ["transactions_last_1h","t1_hi","t1_share"], "velocity_24h": ["transactions_last_24h","t24_hi","burst"],
          "amount": ["log_amt","amt_rel_cat","amt_rel_avg"], "spend": ["log_spend","avg_spend","amt_rel_avg"], "age": ["log_age","young"],
          "new_device": ["new_device"], "night": ["night"], "foreign": ["foreign"]}
for g, fs in groups.items(): go("-" + g, lr_f([f for f in N if f not in fs], Cc))
for c in Cc: go("-" + c, lr_f(N, [x for x in Cc if x != c]))
pd.DataFrame(res, columns=["variant", "pr_auc", "pr_auc_tw"]).round(4).to_csv("experiments/ablation.csv", index=False)
