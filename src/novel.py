"""v5 without any velocity information (features and kNN metric)."""
import sys, warnings, numpy as np, pandas as pd, hashlib; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import models
from r9lab import base, embed, class_dists, EMB, screen, TR
VEL = {"transactions_last_24h", "transactions_last_1h", "t1_hi", "t24_hi", "burst", "t1_share", "t1_ge4_nb", "avg_spend", "amt_rel_avg", "log_spend"}
EMB_NV = [c for c in EMB if c not in VEL]
def nv(a, b):
    A, B, ya, num = base(a, b); num = [c for c in num if c not in VEL]
    Ea, Eb = embed(A, B, EMB_NV); fa, fb = class_dists(Ea, Eb, ya, 1, 1); la, lb = class_dists(Ea, Eb, ya, 0, 1)
    A["dr"] = np.log1p(fa[:, 0]) - np.log1p(la[:, 0]); B["dr"] = np.log1p(fb[:, 0]) - np.log1p(lb[:, 0])
    return models.lr_pipeline(0.1, num + ["dr"], models.LR_CAT).fit(A[models.LR_CAT + num + ["dr"]], ya).predict_proba(B[models.LR_CAT + num + ["dr"]])[:, 1]
if __name__ == "__main__":
    print("EMB without velocity:", EMB_NV)
    screen("v5 NO velocity (local, train regime)", nv)
    te = pd.read_csv("test.csv"); p = nv(TR, te)
    f = "submissions/shift/NV_v5_no_velocity.csv"; pd.DataFrame({"id": te.id, "prediction": p}).to_csv(f, index=False)
    from scipy.stats import spearmanr
    w2 = pd.read_csv("submissions/shift/W2_burst-1.0_ndold-0.75.csv").prediction.values
    b = (te.transactions_last_24h >= 16).values
    print("hash", hashlib.sha256(open(f, "rb").read()).hexdigest()[:10], "spearman vs W2", round(spearmanr(p, w2)[0], 4),
          "| young bursts in top120:", int(((te.account_age <= 365).values & b)[np.argsort(-p)[:120]].sum()))
