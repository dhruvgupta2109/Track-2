import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap
from common import cv_splits, run_cv
from r9lab import make, v5_local, TR, Y, W_HN
from exp_r13_union import suraj
from make_small_batch import pool
from exp_r13_fresh import mlp_fp
F = {"v5": make(lambda A, B, y: v5_local(A, B, y)), "k01": make(pool("merchant_category")), "sur": suraj(), "surk": suraj(knn=True), "mlp": mlp_fp}
rk = lambda x: np.vstack([rankdata(r) / len(r) for r in x])
B = {"k01": lambda O: O["k01"], "0.7k01+0.3sur": lambda O: 0.7 * rk(O["k01"]) + 0.3 * rk(O["sur"]),
     "0.45k01+0.4surk+0.15mlp": lambda O: 0.45 * rk(O["k01"]) + 0.4 * rk(O["surk"]) + 0.15 * rk(O["mlp"]),
     "0.4k01+0.15v5+0.3sur+0.15mlp": lambda O: 0.4 * rk(O["k01"]) + 0.15 * rk(O["v5"]) + 0.3 * rk(O["sur"]) + 0.15 * rk(O["mlp"]),
     "0.5k01+0.35sur+0.15mlp": lambda O: 0.5 * rk(O["k01"]) + 0.35 * rk(O["sur"]) + 0.15 * rk(O["mlp"])}
for seed in [99, 314, 4242, 777]:
    sp = cv_splits(Y, n_repeats=2, seed=seed); O = {k: run_cv(f, TR, sp, 2, verbose=False)[0] for k, f in F.items()}
    m = lambda o, w=None: np.mean([ap(Y, o[r], sample_weight=w) for r in range(2)])
    print(f"seed {seed}: " + " | ".join(f"{k} {m(g(O))-m(O['v5']):+.4f}/{m(g(O),W_HN)-m(O['v5'],W_HN):+.4f}" for k, g in B.items()), flush=True)
