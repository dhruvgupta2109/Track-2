import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from r9lab import screen
import augment as G
def F(gen=None, n=2000, w=0.25, pseudo=None, **kw):
    def fp(a, b):
        rng = np.random.default_rng(len(a))
        syn = None
        if gen == "smote": syn = G.smote(a, n, rng)
        elif gen == "jitter": syn = G.jitter(a, n, rng)
        elif gen == "copula": syn = G.copula(a, n, rng)
        elif gen == "smote_ordinary": syn = G.smote(a, n, rng, pool=G.ordinary_pool(a))
        elif gen == "smote+hardlegit":
            syn = pd.concat([G.smote(a, n, rng), G.smote(a, n, rng, pool=G.risky_legit_pool(a), label=0)])
        elif gen == "mix":
            syn = pd.concat([G.smote(a, n // 3, rng), G.copula(a, n // 3, rng), G.smote(a, n // 3, rng, pool=G.ordinary_pool(a))])
        if syn is not None: syn = syn.assign(w=w)
        ps = None
        if pseudo:
            p0 = G.fit_predict(a, b); lo = p0 <= np.quantile(p0, pseudo[0]); hi = p0 >= np.quantile(p0, 1 - pseudo[1])
            ps = pd.concat([b[lo].assign(fraud=0, w=pseudo[2]), b[hi].assign(fraud=1, w=pseudo[2])])
        return G.fit_predict(a, b, syn, ps)
    return fp
screen("aug none (v5 via augment path)", F())
for gen in ["smote", "jitter", "copula", "smote_ordinary", "smote+hardlegit", "mix"]:
    for n, w in [(1000, 0.25), (3000, 0.1), (3000, 0.5)]:
        screen(f"aug {gen} n={n} w={w}", F(gen, n, w))
screen("pseudo: bottom 30% legit, w0.25", F(pseudo=(0.3, 0.0, 0.25)))
screen("pseudo: bottom 50% legit + top 0.2% fraud, w0.25", F(pseudo=(0.5, 0.002, 0.25)))
