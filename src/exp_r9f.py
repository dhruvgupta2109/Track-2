import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
from r9lab import *
from evaluate import evaluate, prevalence_deltas
from common import cv_splits
E1 = [c for c in EMB if c != "amt_rel_avg"] + ["burst"]
def loc(cols):
    def fn(A, B, y):
        Ea, Eb = embed(A, B, cols); fa, fb = class_dists(Ea, Eb, y, 1, 1); la, lb = class_dists(Ea, Eb, y, 0, 1)
        return pd.DataFrame({"dr": np.log1p(fa[:, 0]) - np.log1p(la[:, 0])}), pd.DataFrame({"dr": np.log1p(fb[:, 0]) - np.log1p(lb[:, 0])})
    return fn
v5f, e1f = make(loc(EMB)), make(loc(E1))
for seed in [99, 314, 7, 4242]:
    sp = cv_splits(Y, n_repeats=3, seed=seed)
    a, _ = run_cv(v5f, TR, sp, 3, verbose=False); b, _ = run_cv(e1f, TR, sp, 3, verbose=False)
    print(f"fresh seed {seed}: E1-v5 d_std {np.mean([ap(Y,b[r])-ap(Y,a[r]) for r in range(3)]):+.4f} d_hnW {np.mean([ap(Y,b[r],sample_weight=W_HN)-ap(Y,a[r],sample_weight=W_HN) for r in range(3)]):+.4f}", flush=True)
o, v = evaluate("v6_E1", e1f, quick=False, ref="v5_prod")
print("prev deltas vs v5:", {k: round(x, 4) for k, x in prevalence_deltas(o, O5).items()})
