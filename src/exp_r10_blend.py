import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap
from sklearn.neural_network import MLPClassifier
from deep import design_v5
from r9lab import TR, Y, SPLITS, W_HN, O5
reps = 2; folds = SPLITS[: 5 * reps]
def skmlp(Xa, ya, Xb, seeds=5):
    return np.mean([MLPClassifier((64, 32), alpha=1e-2, max_iter=200, early_stopping=True, validation_fraction=0.125, n_iter_no_change=10, random_state=s)
                    .fit(Xa, ya).predict_proba(Xb)[:, 1] for s in range(seeds)], 0)
if __name__ == "__main__":
    o = np.zeros((reps, len(Y)))
    for i, (a, b) in enumerate(folds):
        Xa, Xb, ya, _ = design_v5(TR.iloc[a], TR.iloc[b]); o[i // 5, b] = skmlp(Xa, ya, Xb)
    np.save("experiments/oofscreen_skmlp_v5design.npy", o)
    P = {"skmlp": o, "mlp": np.load("experiments/oofscreen_mlp_small10.npy"), "tabm": np.load("experiments/oofscreen_tabm32_reg.npy"), "nca": np.load("experiments/oofscreen_nca_mlp32_noplr.npy")}
    rk = lambda x: np.vstack([rankdata(r) / len(r) for r in x]); f = lambda q, w=None: np.mean([ap(Y, q[r], sample_weight=w) for r in range(reps)])
    b5 = (f(O5[:reps]), f(O5[:reps], W_HN)); print("skmlp(v5 design) standalone", round(f(o), 4))
    def show(nm, q): print(f"{nm:45s} d_std {f(q)-b5[0]:+.4f} d_hnW {f(q, W_HN)-b5[1]:+.4f}", flush=True)
    for w in [0.1, 0.15, 0.2]: show(f"v5 + {w} skmlp", (1 - w) * rk(O5[:reps]) + w * rk(o))
    show("v5 0.8 + skmlp 0.1 + nca 0.1", 0.8 * rk(O5[:reps]) + 0.1 * rk(o) + 0.1 * rk(P["nca"]))
    show("v5 0.8 + skmlp 0.1 + tabm 0.1", 0.8 * rk(O5[:reps]) + 0.1 * rk(o) + 0.1 * rk(P["tabm"]))
    show("v5 0.7 + skmlp .1 + tabm .1 + nca .1", 0.7 * rk(O5[:reps]) + 0.1 * (rk(o) + rk(P["tabm"]) + rk(P["nca"])))
    show("v5 0.7 + mean-rank(4 deep) 0.3", 0.7 * rk(O5[:reps]) + 0.3 * np.mean([rk(v) for v in P.values()], 0))
    show("v5 0.8 + mean-rank(4 deep) 0.2", 0.8 * rk(O5[:reps]) + 0.2 * np.mean([rk(v) for v in P.values()], 0))
