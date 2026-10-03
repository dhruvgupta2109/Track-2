import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
from evaluate import evaluate, prevalence_deltas, Y, W_HN, TR
from common import cv_splits, run_cv
import build_v5, models
from build_v5 import fit_v5, predict_v5
o5 = np.load("experiments/oof5_v5_prod.npy")
base_num = list(models.LR_NUM_V4)
def v5_with(extra_num=(), C=0.1, knn=True):
    def fp(a, b):
        old = models.LR_NUM_V4; models.LR_NUM_V4 = base_num + list(extra_num)
        try:
            bb = fit_v5(a, knn); bb["model"].set_params(logisticregression__C=C)
            return predict_v5(bb, b) if C == 0.1 else _refit(bb, a, b, knn)
        finally: models.LR_NUM_V4 = old
    return fp
def _refit(bb, a, b, knn):
    raise NotImplementedError
def bag(nb=5, frac=0.8):
    def fp(a, b):
        rng = np.random.default_rng(len(a)); ps = []
        for k in range(nb):
            idx = rng.choice(len(a), int(frac * len(a)), replace=False)
            ps.append(predict_v5(fit_v5(a.iloc[idx], True), b))
        return np.mean(ps, 0)
    return fp
def rep(name, fp, quick=False):
    o, v = evaluate(name, fp, quick=quick, ref="v5_prod")
    print("    prev deltas vs v5:", {k: round(x, 4) for k, x in prevalence_deltas(o, o5, prevs=(0.0075, 0.01, 0.0177), n=60).items()}, flush=True)
    return o
rep("v5_t1lin", v5_with(["transactions_last_1h", "t1_hi", "t1_share"]))
rep("v5_t1raw", v5_with(["transactions_last_1h"]), quick=True)
rep("v5_bag5", bag(5), quick=True)
