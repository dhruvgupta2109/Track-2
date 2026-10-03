import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap
from common import cv_splits
from exp_r9_nn import design
from exp_r9_skmlp import skmlp
from r9lab import *
E1 = [c for c in EMB if c != "amt_rel_avg"] + ["burst"]
def lr_on(Xa, Xb, A, B, ya, num): return fit_lr(A, B, ya, num)
rk = lambda x: rankdata(x) / len(x)
for seed in [99, 314, 4242]:
    sp = cv_splits(Y, n_repeats=2, seed=seed); o5 = np.zeros((2, len(Y))); o6 = np.zeros((2, len(Y))); om = np.zeros((2, len(Y)))
    for i, (ia, ib) in enumerate(sp):
        a, b = TR.iloc[ia], TR.iloc[ib]
        o5[i // 5, ib] = make(lambda A, B, y: v5_local(A, B, y))(a, b)
        Xa, Xb, ya, A, B, Ea, Eb = design(a, b); _, _, _, num = base(a.iloc[:50], b.iloc[:5])
        o6[i // 5, ib] = fit_lr(A, B, ya, num + ["dr"]); om[i // 5, ib] = skmlp(Xa, ya, Xb)
    bl = np.vstack([0.9 * rk(o6[r]) + 0.1 * rk(om[r]) for r in range(2)])
    f = lambda o, w=None: np.mean([ap(Y, o[r], sample_weight=w) for r in range(2)])
    print(f"seed {seed}: v6-v5 d_std {f(o6)-f(o5):+.4f} d_hnW {f(o6,W_HN)-f(o5,W_HN):+.4f} | v6+10%MLP - v5 d_std {f(bl)-f(o5):+.4f} d_hnW {f(bl,W_HN)-f(o5,W_HN):+.4f}", flush=True)
