"""Explainable Boosting Machine (bagged boosted GAM / GA2M) on the exact 5x5 folds (experiments/splits_5x5.npz). Runs in .venv_ebm.
OOF saved as experiments/oof5_ebm_<cfg>.npy (5, 20000) for the blend harness in the main env."""
import sys, os, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from features_prod import make_features
from constants import fit_constants
from interpret.glassbox import ExplainableBoostingClassifier as EBM

TR = pd.read_csv("train.csv"); Y = TR.fraud.values
S = np.load("experiments/splits_5x5.npz"); SPL = [(S[f"tr{i}"], S[f"va{i}"]) for i in range(25)]
NUM = ["transaction_amount", "transaction_hour", "transactions_last_24h", "spend_last_24h", "account_age", "new_device", "transactions_last_1h",
       "amt_rel_cat", "avg_spend", "amt_rel_avg"]
CAT = ["merchant_category", "country", "transaction_channel"]
CFG = {"gam": dict(interactions=0, outer_bags=16),
       "gam_smooth": dict(interactions=0, outer_bags=16, smoothing_rounds=500, min_samples_leaf=20, learning_rate=0.005),
       "ga2m3": dict(interactions=3, outer_bags=16),
       "ga2m10": dict(interactions=10, outer_bags=16)}


def frame(a, b):
    K = fit_constants(a); A, B = make_features(a, K), make_features(b, K)
    XA = A[NUM + CAT].copy(); XB = B[NUM + CAT].copy()
    for c in CAT: XA[c] = XA[c].astype(str); XB[c] = XB[c].astype(str)
    return XA, XB


def fit_predict(a, b, cfg, seed=0):
    XA, XB = frame(a, b)
    m = EBM(feature_types=["continuous"] * 5 + ["nominal"] + ["continuous"] * 4 + ["nominal"] * 3, random_state=seed, n_jobs=-2, **CFG[cfg])
    m.fit(XA, a.fraud.values); return m.predict_proba(XB)[:, 1], m


if __name__ == "__main__":
    for cfg in sys.argv[1].split(","):
        path = f"experiments/oof5_ebm_{cfg}.npy"; reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        oof = np.load(path) if os.path.exists(path) else np.full((5, len(Y)), np.nan)
        t = time.time()
        for i, (a, b) in enumerate(SPL[:reps * 5]):
            r = i // 5
            if not np.isnan(oof[r, b]).any(): continue
            oof[r, b], _ = fit_predict(TR.iloc[a], TR.iloc[b], cfg); np.save(path, oof)
        from sklearn.metrics import average_precision_score as ap
        print(f"EBM {cfg}: " + " ".join(f"rep{r} {ap(Y, oof[r]):.4f}" for r in range(reps) if not np.isnan(oof[r]).any()) + f"  ({time.time() - t:.0f}s)", flush=True)
