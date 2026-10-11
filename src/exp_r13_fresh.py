import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap
from common import cv_splits, run_cv
from r9lab import make, v5_local, TR, Y, W_HN
from exp_r13_union import suraj
from deep import design_v5
from exp_r10_blend import skmlp
def mlp_fp(a, b):
    Xa, Xb, ya, _ = design_v5(a, b); return skmlp(Xa, ya, Xb)
F = {"v5": make(lambda A, B, y: v5_local(A, B, y)), "sur": suraj(), "surk": suraj(knn=True), "mlp": mlp_fp}
rk = lambda x: np.vstack([rankdata(r) / len(r) for r in x])
B = {"0.7v5+0.3sur": lambda O: 0.7 * rk(O["v5"]) + 0.3 * rk(O["sur"]), "0.5v5+0.5surk": lambda O: 0.5 * rk(O["v5"]) + 0.5 * rk(O["surk"]),
     "0.6v5+0.4surk": lambda O: 0.6 * rk(O["v5"]) + 0.4 * rk(O["surk"]),
     "0.45v5+0.4surk+0.15mlp": lambda O: 0.45 * rk(O["v5"]) + 0.4 * rk(O["surk"]) + 0.15 * rk(O["mlp"]),
     "0.4v5+0.3surk+0.15sur+0.15mlp": lambda O: 0.4 * rk(O["v5"]) + 0.3 * rk(O["surk"]) + 0.15 * rk(O["sur"]) + 0.15 * rk(O["mlp"])}
if __name__ == "__main__":
  for seed in [99, 314, 4242]:
      sp = cv_splits(Y, n_repeats=2, seed=seed); O = {k: run_cv(f, TR, sp, 2, verbose=False)[0] for k, f in F.items()}
      m = lambda o, w=None: np.mean([ap(Y, o[r], sample_weight=w) for r in range(2)])
      print(f"seed {seed}: " + " | ".join(f"{k} {m(g(O))-m(O['v5']):+.4f}/{m(g(O),W_HN)-m(O['v5'],W_HN):+.4f}" for k, g in B.items()), flush=True)
