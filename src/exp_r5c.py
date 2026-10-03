import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np
from sklearn.metrics import average_precision_score as ap
from common import cv_splits, run_cv
from evaluate import evaluate, Y, W_HN, TR, boot_diff
import r5lab
evaluate("v5a_burstnight_spike", r5lab.F(["burst_night", "spike_avg"]), ref="v4")
evaluate("v5b_burstnight", r5lab.F(["burst_night"]), ref="v4")
# fresh-seed confirmation (seeds never used for selection)
for seed in [99, 314]:
    sp = cv_splits(Y, n_repeats=4, seed=seed); res = {}
    for n, add in [("v4", []), ("v5b", ["burst_night"]), ("v5a", ["burst_night", "spike_avg"])]:
        o, _ = run_cv(r5lab.F(add), TR, sp, 4, verbose=False); res[n] = o
    for n in ["v5b", "v5a"]:
        d_std = np.mean([ap(Y, res[n][r]) - ap(Y, res["v4"][r]) for r in range(4)]); d_hn = np.mean([ap(Y, res[n][r], sample_weight=W_HN) - ap(Y, res["v4"][r], sample_weight=W_HN) for r in range(4)])
        print(f"fresh seed {seed}: {n} - v4: d_std {d_std:+.4f}  d_hnW {d_hn:+.4f}", flush=True)
