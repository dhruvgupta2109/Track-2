import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from r9lab import screen
import augment as G
def MP(gen, alpha, n=2000):
    """mass-preserving: real frauds weight (1-alpha); synthetic frauds share alpha * n_real_fraud total weight."""
    def fp(a, b):
        rng = np.random.default_rng(len(a)); nf = int(a.fraud.sum())
        syn = {"smote": lambda: G.smote(a, n, rng), "jitter": lambda: G.jitter(a, n, rng, s=0.1), "copula": lambda: G.copula(a, n, rng)}[gen]()
        syn = syn.assign(w=alpha * nf / len(syn))
        a2 = a.copy(); orig = G.fit_predict
        # real fraud down-weight via a pseudo frame trick: pass a with weights through syn mechanism
        real = a2.assign(w=np.where(a2.fraud == 1, 1 - alpha, 1.0))
        return _fit(a, b, real, syn)
    return fp
def _fit(a, b, real, syn):
    # reuse augment.fit_predict but with custom real weights: emulate by passing real rows as 'pseudo' with weights and a zero-weight base
    import models, r6lab
    from features_prod import make_features; from constants import fit_constants
    from r9lab import embed, class_dists, EMB
    K = fit_constants(a); ya = a.fraud.values; full = pd.concat([real, syn], ignore_index=True); nreal = len(a)
    Ar, Af, B = make_features(a, K), make_features(full, K), make_features(b, K); num = list(models.LR_NUM_V4)
    for D in (Af, B): D["age_lt120"] = (D.account_age < 120).astype(float).where(D.account_age.notna())
    num.append("age_lt120"); _, cf = r6lab.ctx_amount(Ar, Af, ya); _, cb = r6lab.ctx_amount(Ar, B, ya)
    Af[cf.columns] = cf.values; B[cb.columns] = cb.values; num += list(cf.columns)
    Er, Ef = embed(Ar, Af, EMB); _, Eb = embed(Ar, B, EMB); ns = len(full) - nreal
    dfr, dfx = class_dists(Er, np.r_[Ef[nreal:], Eb], ya, 1, 2); dlr, dlx = class_dists(Er, np.r_[Ef[nreal:], Eb], ya, 0, 1)
    Af["dr"] = np.r_[np.log1p(dfr[:, 0]) - np.log1p(dlr[:, 0]), np.log1p(dfx[:ns, 1]) - np.log1p(dlx[:ns, 0])]
    B["dr"] = np.log1p(dfx[ns:, 0]) - np.log1p(dlx[ns:, 0]); num.append("dr")
    m = models.lr_pipeline(0.1, num, models.LR_CAT); m.fit(Af[models.LR_CAT + num], full.fraud.values, logisticregression__sample_weight=full.w.values)
    return m.predict_proba(B[models.LR_CAT + num])[:, 1]
for gen in ["smote", "jitter", "copula"]:
    for alpha in [0.1, 0.25, 0.5]:
        screen(f"mass-preserving {gen} alpha={alpha}", MP(gen, alpha))
def PS(q, w):
    def fp(a, b):
        p0 = G.fit_predict(a, b); lo = p0 <= np.quantile(p0, q)
        return G.fit_predict(a, b, None, b[lo].assign(fraud=0, w=w))
    return fp
for q, w in [(0.3, 0.5), (0.3, 1.0), (0.5, 0.25), (0.7, 0.25)]:
    screen(f"pseudo legit bottom {q} w={w}", PS(q, w))
