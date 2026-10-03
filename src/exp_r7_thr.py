import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from evaluate import evaluate, prevalence_deltas
import models
from features_prod import make_features; from constants import fit_constants
ref = np.load("experiments/oof5_v4.npy")
def F(extra_fn, add, drop=()):
    def fp(a, b):
        K = fit_constants(a); A = extra_fn(make_features(a, K)); B = extra_fn(make_features(b, K))
        num = [c for c in models.LR_NUM_V4 if c not in drop] + add
        return models.lr_pipeline(0.1, num, models.LR_CAT).fit(A[models.LR_CAT + num], a.fraud).predict_proba(B[models.LR_CAT + num])[:, 1]
    return fp
def step(col, t, name, op="gt"):
    def f(d):
        x = d[col]; d[name] = ((x > t) if op == "gt" else (x < t)).astype(float).where(x.notna()); return d
    return f
def run(name, fp):
    o, v = evaluate(name, fp, quick=True, ref="v4")
    print("     prev deltas:", {k: round(x, 4) for k, x in prevalence_deltas(o, ref, n=60).items()}, flush=True)
for t in [300, 500, 750, 1000]:
    run(f"r7_amt_gt{t}", F(step("transaction_amount", t, "amt_step"), ["amt_step"]))
for t in [120, 180, 365]:
    run(f"r7_age_lt{t}", F(step("account_age", t, "age_step", "lt"), ["age_step"]))
run("r7_t1_ge3_all", F(step("transactions_last_1h", 2.5, "t1_step"), ["t1_step"]))
