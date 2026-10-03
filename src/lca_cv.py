"""CV check: latent-class (mixture of naive Bayes) generative classifier vs v5. If the generator is class-conditional with
independent features inside fraud/legit types, this should be near-Bayes-optimal."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "src")
from lca_shift import discretize, fit_lca, class_loglik
tr = pd.read_csv("train.csv"); y = tr.fraud.values; o5 = np.load("experiments/oof5_v5_prod.npy")

def lca_scores(a_idx, b_idx, Kf, Kl, iters=150, seed=0):
    ca = discretize(tr.iloc[a_idx], tr.iloc[a_idx]); cb = discretize(tr.iloc[a_idx], tr.iloc[b_idx])
    sub = lambda codes, m: [(c[m], L) for c, L in codes]; ya = y[a_idx]
    lwf, thf = fit_lca(sub(ca, ya == 1), Kf, iters, seed); lwl, thl = fit_lca(sub(ca, ya == 0), Kl, iters, seed)
    f = class_loglik(cb, thf) + lwf; l = class_loglik(cb, thl) + lwl
    lse = lambda M: M.max(1) + np.log(np.exp(M - M.max(1, keepdims=True)).sum(1))
    return lse(f) - lse(l)

if __name__ == "__main__":
    for Kf, Kl in [(1, 1), (1, 8), (3, 8), (4, 16), (6, 24)]:
        res = []; resv = []
        for rep in range(2):
            p = np.zeros(len(y))
            for a, b in StratifiedKFold(5, shuffle=True, random_state=2026 + rep).split(tr, y): p[b] = lca_scores(a, b, Kf, Kl)
            res.append(ap(y, p)); resv.append(ap(y, o5[rep]))
        print(f"Kf={Kf} Kl={Kl}: LCA AP {np.mean(res):.4f}  (v5 OOF on its own splits {np.mean(resv):.4f})", flush=True)
