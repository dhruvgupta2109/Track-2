import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np
from evaluate import evaluate, prevalence_deltas
import r6lab
ref = np.load("experiments/oof5_v4.npy")
def run(name, fp):
    o, v = evaluate(name, fp, quick=True, ref="v4")
    print("     prevalence deltas vs v4:", {k: round(x, 4) for k, x in prevalence_deltas(o, ref).items()}, flush=True)
run("r6_knn", r6lab.F(gens=("knn",)))
run("r6_knn_rateonly", r6lab.F(gens=("knn",), keep=["knn_rate"]))
run("r6_iforest", r6lab.F(gens=("iforest",)))
run("r6_legitdens", r6lab.F(gens=("legitdens",)))
run("r6_woe", r6lab.F(gens=("woe",)))
run("r6_ctxamt", r6lab.F(gens=("ctxamt",)))
run("r6_moe_newdevice", lambda a, b: r6lab.moe(a, b, "new_device"))
run("r6_moe_young", lambda a, b: r6lab.moe(a, b, "young"))
run("r6_moe_foreign", lambda a, b: r6lab.moe(a, b, "foreign"))
