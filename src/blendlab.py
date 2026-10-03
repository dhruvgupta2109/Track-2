"""Base-model diversity lab (after friend's Q30: 0.7 v5 + 0.3 pairwise-interaction LR gave +0.0077 public).
Fold-safe OOF on evaluate.SPLITS (5x5, seed 2026) for candidate complements, then logit-blend screens vs v5 OOF."""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, "src")
from evaluate import TR, Y, SPLITS
from common import run_cv

CAT = ["merchant_category", "country", "transaction_channel"]
CORE = ["log_amt", "log_spend", "log_age", "transactions_last_24h", "transactions_last_1h", "new_device", "amt_vs_cat", "foreign", "young", "burst", "night"]


def poly_consts(a, m=20.0):
    y = a.fraud.values; pi = y.mean(); prior = np.log(pi / (1 - pi)); risk = {}
    for c in CAT:
        g = a.assign(_c=a[c].fillna("NA")).groupby("_c").fraud.agg(["sum", "count"])
        p = (g["sum"] + m * pi) / (g["count"] + m); risk[c] = np.log(p / (1 - p)).to_dict()
    return {"cat_med_log": np.log1p(a.transaction_amount).groupby(a.merchant_category).median().to_dict(), "risk": risk, "prior": prior}


def poly_frame(df, PC):
    d = pd.DataFrame(index=df.index)
    d["log_amt"] = np.log1p(df.transaction_amount); d["log_spend"] = np.log1p(df.spend_last_24h); d["log_age"] = np.log1p(df.account_age)
    d["transactions_last_24h"] = df.transactions_last_24h; d["transactions_last_1h"] = df.transactions_last_1h; d["new_device"] = df.new_device
    d["amt_vs_cat"] = d.log_amt - df.merchant_category.map(PC["cat_med_log"]).astype(float)
    d["foreign"] = (df.country != "SG").astype(float).where(df.country.notna())
    d["young"] = (df.account_age < 180).astype(float).where(df.account_age.notna())
    d["burst"] = (df.transactions_last_24h >= 16).astype(float).where(df.transactions_last_24h.notna())
    d["night"] = df.transaction_hour.isin([0, 1, 2, 3, 4, 22, 23]).astype(float)
    for c in CAT:
        d[c] = df[c]; d[c + "_risk"] = df[c].fillna("NA").map(PC["risk"][c]).astype(float).fillna(PC["prior"])
    return d


NUMP = CORE + [c + "_risk" for c in CAT]


def poly_pipe(C=0.003, degree=2, num=NUMP):
    numt = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PolynomialFeatures(degree, include_bias=False), StandardScaler())
    cat = make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(handle_unknown="ignore"))
    return make_pipeline(ColumnTransformer([("num", numt, num), ("cat", cat, CAT)]), LogisticRegression(C=C, max_iter=5000))


def fp_poly(C=0.003, degree=2, extra_hour=False):
    def f(a, b):
        PC = poly_consts(a); A, B = poly_frame(a, PC), poly_frame(b, PC); num = list(NUMP)
        if extra_hour:
            for D, s in ((A, a), (B, b)): D["hc"] = s.transaction_hour.astype(str)
        m = poly_pipe(C, degree, num)
        if extra_hour:
            m.steps[0][1].transformers.append(("h", make_pipeline(OneHotEncoder(handle_unknown="ignore")), ["hc"]))
        m.fit(A, a.fraud.values); return m.predict_proba(B)[:, 1]
    return f


def fp_rbf(C=0.01, gamma=0.05, n=600):
    def f(a, b):
        PC = poly_consts(a); A, B = poly_frame(a, PC), poly_frame(b, PC)
        numt = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Nystroem(gamma=gamma, n_components=n, random_state=0), StandardScaler())
        cat = make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(handle_unknown="ignore"))
        m = make_pipeline(ColumnTransformer([("num", numt, NUMP), ("cat", cat, CAT)]), LogisticRegression(C=C, max_iter=5000))
        m.fit(A, a.fraud.values); return m.predict_proba(B)[:, 1]
    return f


def fp_spline(C=0.01, k=5):
    def f(a, b):
        PC = poly_consts(a); A, B = poly_frame(a, PC), poly_frame(b, PC)
        numt = make_pipeline(SimpleImputer(strategy="median"), SplineTransformer(n_knots=k, degree=3), StandardScaler(), PolynomialFeatures(2, interaction_only=True, include_bias=False), StandardScaler())
        cat = make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(handle_unknown="ignore"))
        m = make_pipeline(ColumnTransformer([("num", numt, NUMP), ("cat", cat, CAT)]), LogisticRegression(C=C, max_iter=5000))
        m.fit(A, a.fraud.values); return m.predict_proba(B)[:, 1]
    return f


lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
O5 = np.load("experiments/oof5_v5_prod.npy")


def screen(oof, name, reps=None):
    reps = reps or range(len(oof))
    base = np.mean([ap(Y, O5[r]) for r in reps]); own = np.mean([ap(Y, oof[r]) for r in reps])
    out = [f"{name}: own {own:.4f} | v5 {base:.4f}"]
    for w in (0.2, 0.3, 0.4, 0.5):
        s = [ap(Y, (1 - w) * lg(O5[r]) + w * lg(oof[r])) for r in reps]
        out.append(f"w{w}: {np.mean(s):.4f} ({np.mean(s) - base:+.4f}, min rep {min(np.array(s) - [ap(Y, O5[r]) for r in reps]):+.4f})")
    print("  ".join(out), flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "poly"
    cfg = {"poly": lambda: fp_poly(), "poly_C0.01": lambda: fp_poly(0.01), "poly_C0.001": lambda: fp_poly(0.001), "poly_hour": lambda: fp_poly(extra_hour=True),
           "poly3": lambda: fp_poly(0.001, 3), "rbf": lambda: fp_rbf(), "rbf_g0.02": lambda: fp_rbf(0.01, 0.02), "rbf_C0.003": lambda: fp_rbf(0.003, 0.05),
           "spline": lambda: fp_spline()}
    for k in which.split(","):
        path = f"experiments/oof5_bl_{k}.npy"
        if os.path.exists(path): oof = np.load(path)
        else:
            oof, _ = run_cv(cfg[k](), TR, SPLITS, 5, verbose=False); np.save(path, oof)
        screen(oof, k)
