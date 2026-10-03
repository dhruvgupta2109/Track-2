import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np
from evaluate import evaluate, prevalence_deltas
import r6lab
ref = np.load("experiments/oof5_v4.npy")
def run(name, fp, quick=True):
    o, v = evaluate(name, fp, quick=quick, ref="v4")
    print("     prevalence deltas vs v4:", {k: round(x, 4) for k, x in prevalence_deltas(o, ref).items()}, flush=True)
run("r6_knn_dfraud", r6lab.F(gens=("knn",), keep=["d_fraud"]))
run("r6_knn_dlegit", r6lab.F(gens=("knn",), keep=["d_legit"]))
run("r6_knn_dratio", r6lab.F(gens=("knn",), keep=["d_ratio"]))
run("r6_knn_dfraud_dlegit", r6lab.F(gens=("knn",), keep=["d_fraud", "d_legit"]))
run("r6_knn_full", r6lab.F(gens=("knn",)), quick=False)
