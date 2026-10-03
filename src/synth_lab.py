"""Program synthesis (generator-structure search) inside each CV fold.
Library of ~270 candidate programs: step thresholds at round numbers on every numeric field, hinge terms, hour/category one-hots,
and pairwise ANDs of 15 risk flags. Structure is selected by L1 logistic regression whose strength is chosen by inner 3-fold CV on the
outer-training part only (AP), then refit (optionally L2 'relaxed' refit on the selected programs). Stability: which programs are
rediscovered across independent folds."""
import sys, os, json, itertools, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, "src")
from evaluate import TR, Y, SPLITS

TH = {"amt": [10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000],
      "rel": [0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20], "spend": [0.5, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000],
      "avg": [10, 20, 50, 100, 200, 500, 1000], "ravg": [0.25, 0.5, 1, 2, 3, 5, 10, 20],
      "age": [7, 14, 30, 60, 90, 120, 180, 270, 365, 500, 730, 1000, 1500, 2000, 3000], "t24": [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
      "t1": [1, 2, 3, 4, 5, 6, 7, 8], "share": [0.1, 0.2, 0.3, 0.5, 0.75, 0.99]}
CATS = ["merchant_category", "country", "transaction_channel"]


def raw(df, cat_med):
    amt = df.transaction_amount.astype(float); t24 = df.transactions_last_24h.astype(float); sp = df.spend_last_24h.astype(float)
    avg = sp / t24.clip(lower=1)
    return {"amt": amt, "rel": amt / df.merchant_category.map(cat_med).astype(float), "spend": sp, "avg": avg, "ravg": amt / (avg + 1),
            "age": df.account_age.astype(float), "t24": t24, "t1": df.transactions_last_1h.astype(float), "share": df.transactions_last_1h / t24.clip(lower=1)}


def library(df, cat_med, levels):
    R = raw(df, cat_med); F = {}
    for k, ths in TH.items():
        x = R[k]
        for t in ths:
            F[f"{k}>={t}" if k in ("t24", "t1") else f"{k}>{t}"] = ((x >= t) if k in ("t24", "t1") else (x > t)).astype(float).where(x.notna())
        F[f"log_{k}"] = np.log1p(x.clip(lower=0))
    h = df.transaction_hour
    for v in range(24): F[f"hour={v}"] = (h == v).astype(float)
    for c in CATS:
        for v in levels[c]: F[f"{c}={v}"] = (df[c] == v).astype(float).where(df[c].notna())
    nd = df.new_device.astype(float); F["new_device"] = nd
    flags = {"nd": nd, "night": h.isin([23, 0, 1, 2, 3, 4]).astype(float), "foreign": (df.country != "SG").astype(float).where(df.country.notna()),
             "age<90": (R["age"] < 90).astype(float).where(R["age"].notna()), "age<180": (R["age"] < 180).astype(float).where(R["age"].notna()),
             "age<365": (R["age"] < 365).astype(float).where(R["age"].notna()), "burst": (R["t24"] >= 16).astype(float).where(R["t24"].notna()),
             "t1>=3": (R["t1"] >= 3).astype(float).where(R["t1"].notna()), "amt>500": (R["amt"] > 500).astype(float), "amt>1000": (R["amt"] > 1000).astype(float),
             "rel>3": (R["rel"] > 3).astype(float).where(R["rel"].notna()), "ecom": (df.transaction_channel == "ecommerce").astype(float).where(df.transaction_channel.notna()),
             "bank": (df.transaction_channel == "bank_transfer").astype(float).where(df.transaction_channel.notna()),
             "card_present": (df.transaction_channel == "card_present").astype(float).where(df.transaction_channel.notna()),
             "riskycat": df.merchant_category.isin(["cash_transfer", "electronics", "luxury"]).astype(float).where(df.merchant_category.notna())}
    for (a, x), (b, y) in itertools.combinations(flags.items(), 2): F[f"{a}&{b}"] = x * y
    X = pd.DataFrame(F, index=df.index)
    return X


class Synth:
    def __init__(self, Cs=(0.003, 0.005, 0.01, 0.02, 0.05, 0.1), relaxed=False, l2C=0.1):
        self.Cs, self.relaxed, self.l2C = Cs, relaxed, l2C

    def _prep(self, df, fit=False):
        if fit:
            self.cat_med = df.groupby("merchant_category").transaction_amount.median().to_dict()
            self.levels = {c: sorted(df[c].dropna().unique()) for c in CATS}
        X = library(df, self.cat_med, self.levels)
        if fit: self.med = X.median(); self.sc = StandardScaler().fit(X.fillna(self.med))
        return self.sc.transform(X.fillna(self.med))

    def _fit_C(self, X, y, C):
        return LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=3000).fit(X, y)

    def fit(self, df):
        y = df.fraud.values; X = self._prep(df, fit=True); self.names = list(library(df.head(5), self.cat_med, self.levels).columns)
        inner = StratifiedKFold(3, shuffle=True, random_state=7); sc = {C: [] for C in self.Cs}
        for a, b in inner.split(X, y):
            for C in self.Cs: sc[C].append(ap(y[b], self._fit_C(X[a], y[a], C).decision_function(X[b])))
        self.C = max(self.Cs, key=lambda C: np.mean(sc[C])); self.inner = {C: float(np.mean(v)) for C, v in sc.items()}
        m = self._fit_C(X, y, self.C); self.sel = np.flatnonzero(m.coef_[0] != 0)
        self.m = LogisticRegression(C=self.l2C, max_iter=5000).fit(X[:, self.sel], y) if self.relaxed else m
        return self

    def predict(self, df):
        X = self._prep(df); X = X[:, self.sel] if self.relaxed else X
        return self.m.predict_proba(X)[:, 1]


if __name__ == "__main__":
    relaxed = len(sys.argv) > 1 and sys.argv[1] == "relaxed"; reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    tag = "synth_relaxed" if relaxed else "synth_l1"; path = f"experiments/oof5_{tag}.npy"
    oof = np.load(path) if os.path.exists(path) else np.full((5, len(Y)), np.nan); chosen = []
    for i, (a, b) in enumerate(SPLITS[:reps * 5]):
        r = i // 5
        if not np.isnan(oof[r, b]).any(): continue
        s = Synth(relaxed=relaxed).fit(TR.iloc[a]); oof[r, b] = s.predict(TR.iloc[b]); np.save(path, oof)
        chosen.append({"fold": i, "C": s.C, "n_selected": int(len(s.sel)), "selected": [s.names[j] for j in s.sel]})
        print(f"fold {i}: C={s.C} selected {len(s.sel)} programs, fold AP {ap(Y[b], oof[r, b]):.4f}", flush=True)
    if chosen: json.dump(chosen, open(f"experiments/{tag}_selection.json", "w"), indent=1)
    import arch_eval; arch_eval.evaluate(oof, tag)
