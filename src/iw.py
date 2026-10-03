"""Importance-weighted v5 (covariate-shift adaptation for a misspecified model): sample_weight = w(x)^gamma,
w = test/train density ratio from the label-free adversarial classifier (experiments/testlikeness_train.npy)."""
import sys, warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import models
from r9lab import base, embed, class_dists, EMB, TR, Y
TL = np.load("experiments/testlikeness_train.npy")
def weights(idx, gamma=1.0, clip=10.0):
    r = np.clip(TL[idx], 0.01, 0.99); w = np.minimum(r / (1 - r) * (20000 / 12000), clip) ** gamma
    return w / w.mean()
def iw_fit_predict(a, b, gamma=1.0, clip=10.0, C=0.1):
    A, B, ya, num = base(a, b)
    Ea, Eb = embed(A, B, EMB); fa, fb = class_dists(Ea, Eb, ya, 1, 1); la, lb = class_dists(Ea, Eb, ya, 0, 1)
    A["dr"] = np.log1p(fa[:, 0]) - np.log1p(la[:, 0]); B["dr"] = np.log1p(fb[:, 0]) - np.log1p(lb[:, 0]); num = num + ["dr"]
    w = weights(a.index.values, gamma, clip) if gamma else np.ones(len(a))
    m = models.lr_pipeline(C, num, models.LR_CAT); m.fit(A[models.LR_CAT + num], ya, logisticregression__sample_weight=w)
    return m.predict_proba(B[models.LR_CAT + num])[:, 1]
if __name__ == "__main__":
    import hashlib
    from sklearn.metrics import average_precision_score as ap
    from evaluate import SPLITS, W_COV
    from common import run_cv
    from features_prod import make_features; from constants import fit_constants
    te = pd.read_csv("test.csv"); D = make_features(te, fit_constants(TR)); lg = lambda p: np.log(p / (1 - p))
    v5 = pd.read_csv("submissions/sub_v5_lr_knn.csv").prediction.values
    b = (D.transactions_last_24h >= 16).values; a_ = D.account_age.values; nd = (D.new_device == 1).values
    seg = {"burst": b, "nd_old": nd & (a_ > 365) & ~b, "big&age>1500": (D.amt_rel_cat > 1).values & (a_ > 1500) & ~b & ~(nd & (a_ > 365)),
           "travel": (D.merchant_category == "travel").values, "t24 10-15": ((D.transactions_last_24h >= 10) & (D.transactions_last_24h < 16)).values,
           "foreign": (D.foreign == 1).values, "luxury": (D.merchant_category == "luxury").values, "young<90": a_ < 90, "night": (D.night == 1).values}
    o5 = np.load("experiments/oof5_v5_prod.npy")[:2]
    for g, c in [(0.5, 10.0), (1.0, 10.0), (1.0, 30.0)]:
        oof, _ = run_cv(lambda a_df, b_df: iw_fit_predict(a_df, b_df, g, c), TR, SPLITS[:10], 2, verbose=False)
        f = lambda o, w=None: np.mean([ap(Y, o[r], sample_weight=w) for r in range(2)])
        p = iw_fit_predict(TR, te, g, c); d = lg(p) - lg(v5); d -= np.median(d)
        print(f"IW gamma={g} clip={c}: OOF std {f(oof):.4f} (v5 {f(o5):.4f})  covW {f(oof, W_COV):.4f} (v5 {f(o5, W_COV):.4f})")
        print("   implied test shifts vs v5:", "  ".join(f"{k} {d[m].mean():+.2f}" for k, m in seg.items()), flush=True)
        fn = f"submissions/shift/IW_g{g}_c{int(c)}.csv"; pd.DataFrame({"id": te.id, "prediction": p}).to_csv(fn, index=False)
        print("   hash", hashlib.sha256(open(fn, "rb").read()).hexdigest()[:10])
