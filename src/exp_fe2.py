import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); from common import *; from features import *
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap
from scipy.stats import rankdata
tr, te = load(); y = tr[LABEL].values
R = 5; splits = cv_splits(y, n_repeats=R)
base_nums = ["log_amt","log_spend","log_age","new_device","transactions_last_24h","transactions_last_1h"]
ENG = ["log_amt","amt_rel_cat","log_spend","avg_spend","amt_rel_avg","log_age","young","t1_hi","t24_hi","burst","t1_share","night","foreign"]
def add_inter(d):
    d["nd_young"] = d.new_device * d.young; d["nd_amtrel"] = d.new_device * d.amt_rel_cat
    d["young_amtrel"] = d.young * d.amt_rel_cat; d["t1hi_amt"] = d.t1_hi * d.amt_rel_cat
    return d
INTER = ["nd_young","nd_amtrel","young_amtrel","t1hi_amt"]

def lr_fp_factory(nums, C=0.3, indicator=True, aug=0.0, inter=False, hourcat=True):
    def fp(a, b):
        K = fit_constants(a)
        if aug:
            rng = np.random.default_rng(len(a)); a = pd.concat([a, inject_missing(a, aug, rng)])
        A = make_features(a, K); B = make_features(b, K)
        if inter: A = add_inter(A); B = add_inter(B)
        cats = CATS + (["hc"] if hourcat else [])
        A["hc"] = A.transaction_hour.astype(str); B["hc"] = B.transaction_hour.astype(str)
        ct = ColumnTransformer([("c", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(handle_unknown="ignore")), cats),
                                ("n", make_pipeline(SimpleImputer(strategy="median", add_indicator=indicator), StandardScaler()), nums)])
        m = make_pipeline(ct, LogisticRegression(C=C, max_iter=5000)).fit(A, a[LABEL])
        return m.predict_proba(B)[:, 1]
    return fp
N = base_nums + ENG
exps = {
 "lr_eng_noind": lr_fp_factory(N, indicator=False),
 "lr_eng_noind_C0.05": lr_fp_factory(N, indicator=False, C=0.05),
 "lr_eng_noind_C0.1": lr_fp_factory(N, indicator=False, C=0.1),
 "lr_eng_noind_C1": lr_fp_factory(N, indicator=False, C=1),
 "lr_eng_noind_inter": lr_fp_factory(N + INTER, indicator=False, inter=True),
 "lr_eng_noind_aug": lr_fp_factory(N, indicator=False, aug=0.03),
}
res = {}
for n, fp in exps.items():
    print(n); oof, s = run_cv(fp, tr, splits, R); np.save(f"experiments/oof_{n}.npy", oof)
    _, s2 = run_cv(fp, tr, splits, R, extra_missing=0.017, verbose=False); s["pr_auc_testmiss"] = s2["pr_auc_mean"]
    print("   with test-like missingness:", round(s2["pr_auc_mean"], 4)); log_result(n, s, "FE round 2"); res[n] = oof
# blends of OOF (per repeat)
lgbo = np.load("experiments/oof_lgb_eng_mono.npy")
for n in ["lr_eng_noind"]:
    for w in [0.3, 0.5, 0.7]:
        sc = [ap(y, w * rankdata(res[n][r]) + (1 - w) * rankdata(lgbo[r])) for r in range(R)]
        sp = [ap(y, w * res[n][r] + (1 - w) * lgbo[r]) for r in range(R)]
        print(f"blend {n} w_lr={w}: rank {np.mean(sc):.4f}  prob {np.mean(sp):.4f}")
