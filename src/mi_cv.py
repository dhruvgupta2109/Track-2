"""Missing values: median imputation (current) vs marginalising over plausible values (multiple imputation, average of
predicted probabilities). CV with test-level extra missingness injected into validation folds only."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "src")
from build_v5 import fit_v5, predict_v5
from common import RAW
tr = pd.read_csv("train.csv"); y = tr.fraud.values
COLS = [c for c in RAW if c not in ("transaction_amount", "transaction_hour")]


def mi_predict(b, ref, df, M=12, seed=0):
    """average v5 probability over M draws; each missing cell gets a value from a random reference (train-fold) row
    of the same merchant category when known (keeps amount/spend scale coherent), else from any row."""
    rng = np.random.default_rng(seed); miss = df[COLS].isna(); rows = miss.any(axis=1).values
    out = predict_v5(b, df)
    if not rows.any(): return out
    sub = df[rows].reset_index(drop=True); msub = miss[rows].reset_index(drop=True)
    ref = ref.reset_index(drop=True); bycat = {c: np.where(ref.merchant_category.values == c)[0] for c in ref.merchant_category.dropna().unique()}
    acc = np.zeros(len(sub))
    for m in range(M):
        S = sub.copy()
        for c in COLS:
            idx = np.where(msub[c].values)[0]
            if not len(idx): continue
            pick = np.array([rng.choice(bycat.get(cat, np.arange(len(ref)))) if isinstance(cat, str) else rng.integers(len(ref)) for cat in S.merchant_category.values[idx]])
            v = ref[c].values[pick]
            # fall back to a random non-missing reference value when the donor itself is missing
            bad = pd.isna(v)
            if bad.any():
                pool = ref[c].dropna().values; v[bad] = pool[rng.integers(len(pool), size=bad.sum())]
            S.loc[idx, c] = v
        acc += predict_v5(b, S) / M
    out = out.copy(); out[rows] = acc; return out


if __name__ == "__main__":
    rate = float(sys.argv[1]) if len(sys.argv) > 1 else 0.016   # extra per-column blanks so validation matches test's ~2.2%/col
    res = {"median": [], "MI": [], "clean": []}
    for rep in range(2):
        rng = np.random.default_rng(100 + rep)
        for a, b_ in StratifiedKFold(5, shuffle=True, random_state=2026 + rep).split(tr, y):
            A, B = tr.iloc[a], tr.iloc[b_].copy(); bundle = fit_v5(A)
            pc = predict_v5(bundle, B)
            for c in COLS: B.loc[rng.random(len(B)) < rate, c] = np.nan
            pm = predict_v5(bundle, B); pmi = mi_predict(bundle, A[A.fraud == 0], B)
            res["clean"].append((b_, pc)); res["median"].append((b_, pm)); res["MI"].append((b_, pmi))
        for k in res:
            oof = np.zeros(len(y))
            for idx, p in res[k][-5:]: oof[idx] = p
            print(f"rep {rep} {k:6s} AP {ap(y, oof):.4f}", flush=True)
