import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from r9lab import *
screen("v5 reproduction (kf1 kl1)", make(lambda A, B, y: v5_local(A, B, y)))
for kl in [3, 5, 10, 20]:
    screen(f"asym kf1, kl{kl} mean", make(lambda A, B, y, kl=kl: v5_local(A, B, y, kl=kl, agg="mean")))
screen("asym kf1, kl5 kth", make(lambda A, B, y: v5_local(A, B, y, kl=5, agg="kth")))
# separate d_f and multiple d_l scales as features
def multi(A, B, y):
    Ea, Eb = embed(A, B); fa, fb = class_dists(Ea, Eb, y, 1, 3); la, lb = class_dists(Ea, Eb, y, 0, 10)
    mk = lambda f, l: pd.DataFrame({"dr": np.log1p(f[:, 0]) - np.log1p(l[:, 0]), "dl1": np.log1p(l[:, 0]), "dl10": np.log1p(l.mean(1)), "df1": np.log1p(f[:, 0])})
    return mk(fa, la), mk(fb, lb)
screen("multi: dr + dl1 + dl10mean + df1", make(multi))
def legit_only(A, B, y):
    Ea, Eb = embed(A, B); la, lb = class_dists(Ea, Eb, y, 0, 1)
    return pd.DataFrame({"dl1": np.log1p(la[:, 0])}), pd.DataFrame({"dl1": np.log1p(lb[:, 0])})
screen("legit isolation only (dl1)", make(legit_only))
# channel / new-device / foreign conditional pools
for gcol in ["transaction_channel", "new_device", "foreign", "merchant_category"]:
    def cond(A, B, y, gcol=gcol):
        Ea, Eb = embed(A, B); ga, gb = A[gcol].values, B[gcol].values
        fa, fb = pooled_dists(Ea, Eb, ga, gb, y, 1, 1); la, lb = pooled_dists(Ea, Eb, ga, gb, y, 0, 1)
        return pd.DataFrame({"dr_c": np.log1p(fa[:, 0]) - np.log1p(la[:, 0])}), pd.DataFrame({"dr_c": np.log1p(fb[:, 0]) - np.log1p(lb[:, 0])})
    screen(f"conditional pool {gcol} (replace)", make(cond))
    def cond2(A, B, y, gcol=gcol):
        a1, b1 = v5_local(A, B, y); a2, b2 = cond(A, B, y)
        return pd.concat([a1, a2], axis=1), pd.concat([b1, b2], axis=1)
    screen(f"global + conditional {gcol}", make(cond2))
