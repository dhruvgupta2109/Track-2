import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np
from sklearn.metrics import average_precision_score as ap
from common import cv_splits, run_cv
from r9lab import make, v5_local, TR, Y, W_HN
from make_small_batch import pool, multi_amt_age, with_cash
C = {"v5": make(lambda A, B, y: v5_local(A, B, y)), "k01_merchant_pool": make(pool("merchant_category")), "k04_multimetric": make(multi_amt_age),
     "k02_drop_amt_rel_avg": make(lambda A, B, y: v5_local(A, B, y, cols=[c for c in __import__("r9lab").EMB if c != "amt_rel_avg"])), "k03_add_cash": make(with_cash)}
for seed in [99, 314, 4242]:
    sp = cv_splits(Y, n_repeats=2, seed=seed); O = {k: run_cv(f, TR, sp, 2, verbose=False)[0] for k, f in C.items()}
    m = lambda o, w=None: np.mean([ap(Y, o[r], sample_weight=w) for r in range(2)])
    print(f"seed {seed}: " + "  ".join(f"{k} {m(O[k])-m(O['v5']):+.4f}/{m(O[k],W_HN)-m(O['v5'],W_HN):+.4f}" for k in C if k != "v5"), flush=True)
