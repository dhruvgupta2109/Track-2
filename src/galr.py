"""Generator-aware basis logistic ranker (GALR): unweighted L2 LR on step indicators at plausible synthetic thresholds.
Features (fixed a priori): amount > 100/200/300/500/750/1000/1500/2500; hour one-hot + night flag; account_age < 120/270/365;
transactions_last_1h > 0; transactions_last_24h clipped at 15 + excess above 15; one-hot merchant/country/channel; new_device;
interactions new_device x night, new_device x ecommerce. Compared on evaluate.SPLITS (5x5, seed 2026) against v5 and v5+poly."""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, "src")
from evaluate import TR, Y, SPLITS
from common import run_cv

AMT_T = [100, 200, 300, 500, 750, 1000, 1500, 2500]; AGE_T = [120, 270, 365]
CAT = ["merchant_category", "country", "transaction_channel", "hour_c"]
NIGHT = [23, 0, 1, 2, 3, 4]


def basis(df, inter=True):
    d = pd.DataFrame(index=df.index)
    amt = pd.to_numeric(df.transaction_amount, errors="coerce")
    for t in AMT_T: d[f"amt_gt{t}"] = (amt > t).astype(float).where(amt.notna())
    h = pd.to_numeric(df.transaction_hour, errors="coerce")
    d["night"] = h.isin(NIGHT).astype(float).where(h.notna())
    age = pd.to_numeric(df.account_age, errors="coerce")
    for t in AGE_T: d[f"age_lt{t}"] = (age < t).astype(float).where(age.notna())
    t1 = pd.to_numeric(df.transactions_last_1h, errors="coerce"); d["t1_pos"] = (t1 > 0).astype(float).where(t1.notna())
    t24 = pd.to_numeric(df.transactions_last_24h, errors="coerce"); d["t24_clip"] = t24.clip(upper=15); d["t24_excess"] = (t24 - 15).clip(lower=0)
    nd = pd.to_numeric(df.new_device, errors="coerce"); d["new_device"] = nd
    if inter:
        d["nd_x_night"] = nd * d.night
        d["nd_x_ecom"] = nd * (df.transaction_channel == "ecommerce").astype(float).where(df.transaction_channel.notna())
    for c in CAT[:3]: d[c] = df[c]
    d["hour_c"] = h.round().astype("Int64").astype(str).where(h.notna(), None)
    return d


def pipe(num, C=0.1):
    ct = ColumnTransformer([("c", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(handle_unknown="ignore")), CAT),
                            ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num)])
    return make_pipeline(ct, LogisticRegression(C=C, max_iter=5000))   # class_weight=None (unweighted)


def fp(C=0.1, inter=True):
    def f(a, b):
        A, B = basis(a, inter), basis(b, inter); num = [c for c in A.columns if c not in CAT]
        return pipe(num, C).fit(A, a.fraud.values).predict_proba(B)[:, 1]
    return f


lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
O5 = np.load("experiments/oof5_v5_prod.npy"); PO = np.load("experiments/oof5_bl_poly.npy")
BEST = np.array([0.7 * lg(O5[r]) + 0.3 * lg(PO[r]) for r in range(5)])   # Q30's base blend (v5 + interaction LR) OOF


def per_rep(oof): return np.array([ap(Y, oof[r]) for r in range(len(oof))])


if __name__ == "__main__":
    v5 = per_rep(O5); best = per_rep(BEST)
    print(f"reference on SPLITS: v5 {v5.mean():.4f}  v5+poly(Q30 base) {best.mean():.4f}", flush=True)
    for inter in (True, False):
        for C in (0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3):
            path = f"experiments/oof5_galr_C{C}_{'int' if inter else 'noint'}.npy"
            if os.path.exists(path): oof = np.load(path)
            else: oof, _ = run_cv(fp(C, inter), TR, SPLITS, 5, verbose=False); np.save(path, oof)
            s = per_rep(oof)
            blend = per_rep(np.array([0.8 * BEST[r] + 0.2 * lg(oof[r]) for r in range(5)]))
            print(f"GALR C={C:<5} inter={inter!s:5}: AP {s.mean():.4f} (reps {np.round(s, 4)}) | vs v5 {s.mean() - v5.mean():+.4f} "
                  f"(wins {int((s > v5).sum())}/5) | vs best {s.mean() - best.mean():+.4f} | 0.8 best + 0.2 GALR {blend.mean() - best.mean():+.4f} (wins {int((blend > best).sum())}/5)", flush=True)
