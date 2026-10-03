"""Type-proportion (mixture) shift adaptation with a latent-class model.
Train: fraud rows -> K_f latent classes, legit rows -> K_l latent classes (naive-Bayes within class, categorical/binned features,
missing values marginalised). Test: EM re-estimates only the K_f + K_l class weights (profiles fixed).
Correction c(x) = log-odds_test(x) - log-odds_train(x) under the latent-class model; added to v5's logit."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from features_prod import make_features
from constants import fit_constants

def discretize(tr_raw, X_raw):
    """returns list of (codes array with -1 = missing, n_levels) using cut points from train."""
    out = []
    def qbins(col, q):
        e = np.unique(np.nanquantile(tr_raw[col].astype(float), np.linspace(0, 1, q + 1)[1:-1])); v = X_raw[col].astype(float).values
        c = np.digitize(v, e); c[np.isnan(v)] = -1; out.append((c, len(e) + 1))
    def fixed(col, edges):
        v = X_raw[col].astype(float).values; c = np.digitize(v, edges); c[np.isnan(v)] = -1; out.append((c, len(edges) + 1))
    def cat(col):
        lv = sorted(tr_raw[col].dropna().unique()); m = {l: i for i, l in enumerate(lv)}
        c = X_raw[col].map(m); c = c.fillna(-1).astype(int).values; out.append((c, len(lv)))
    qbins("transaction_amount", 10); qbins("spend_last_24h", 8)
    fixed("account_age", [31, 91, 181, 366, 731, 1501, 2501])
    fixed("transactions_last_24h", [4, 7, 10, 16, 21]); fixed("transactions_last_1h", [1, 2, 3, 5])
    fixed("new_device", [0.5]); fixed("transaction_hour", [5, 8, 13, 18, 21, 23])
    cat("merchant_category"); cat("country"); cat("transaction_channel")
    return out

def class_loglik(codes, theta):
    """theta: list over features of (K, L) log-prob tables. returns (n, K) log-lik, missing marginalised."""
    n = len(codes[0][0]); K = theta[0].shape[0]; ll = np.zeros((n, K))
    for (c, L), T in zip(codes, theta):
        m = c >= 0; ll[m] += T[:, c[m]].T
    return ll

def fit_lca(codes, K, iters=200, seed=0, alpha=1.0):
    rng = np.random.default_rng(seed); n = len(codes[0][0])
    R = rng.dirichlet(np.ones(K), n)                       # responsibilities
    for it in range(iters):
        w = R.mean(0); theta = []
        for c, L in codes:
            T = np.full((K, L), alpha)
            m = c >= 0
            for l in range(L): T[:, l] += R[m & (c == l)].sum(0)
            theta.append(np.log(T / T.sum(1, keepdims=True)))
        ll = class_loglik(codes, theta) + np.log(w)
        R = np.exp(ll - ll.max(1, keepdims=True)); R /= R.sum(1, keepdims=True)
    return np.log(w), theta

def lca_correction(Kf=4, Kl=10, iters=150, seed=0, test_iters=300):
    tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv"); y = tr.fraud.values
    ctr, cte = discretize(tr, tr), discretize(tr, te)
    sub = lambda codes, m: [(c[m], L) for c, L in codes]
    lwf, thf = fit_lca(sub(ctr, y == 1), Kf, iters, seed)
    lwl, thl = fit_lca(sub(ctr, y == 0), Kl, iters, seed)
    pi = y.mean()
    # joint component weights on train
    lw_tr = np.r_[np.log(pi) + lwf, np.log(1 - pi) + lwl]
    LLte = np.c_[class_loglik(cte, thf), class_loglik(cte, thl)]
    # EM on test: re-estimate weights only
    lw = lw_tr.copy()
    for _ in range(test_iters):
        a = LLte + lw; R = np.exp(a - a.max(1, keepdims=True)); R /= R.sum(1, keepdims=True)
        lw_new = np.log(R.mean(0) + 1e-12)
        if np.abs(lw_new - lw).max() < 1e-7: break
        lw = lw_new
    def logodds(LL, lw_):
        f = LL[:, :Kf] + lw_[:Kf]; l = LL[:, Kf:] + lw_[Kf:]
        mf, ml = f.max(1), l.max(1)
        return (mf + np.log(np.exp(f - mf[:, None]).sum(1))) - (ml + np.log(np.exp(l - ml[:, None]).sum(1)))
    c = logodds(LLte, lw) - logodds(LLte, lw_tr)
    info = {"train_w": np.exp(lw_tr), "test_w": np.exp(lw), "test_prior": np.exp(lw[:Kf]).sum()}
    return c, info, te

if __name__ == "__main__":
    import hashlib
    tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv"); D = make_features(te, fit_constants(tr))
    b = (D.transactions_last_24h >= 16).values; a_ = D.account_age.values; nd = (D.new_device == 1).values; t1 = (D.transactions_last_1h >= 3).values
    ndold = nd & (a_ > 365) & ~b
    hv = D.merchant_category.isin(["luxury", "cash_transfer", "electronics", "travel"]).values & (D.transaction_amount >= 2000).values & (D.transaction_amount <= 6000).values & (a_ > 365) & ~b & ~ndold
    G = {"burst_young": (b & (a_ <= 365), "-2.0"), "burst_old": (b & ~(a_ <= 365), "-0.5"), "nd_old": (ndold, "-0.75"), "highvalue_est": (hv, "+0.5"),
         "travel": ((D.merchant_category == "travel").values & ~b & ~ndold & ~hv, "~0"), "nd<=90_fast": (nd & (a_ <= 90) & t1 & ~b, "~0"),
         "nd<180_night_lowvel": (nd & (a_ < 180) & D.transaction_hour.isin([22, 23, 0, 1, 2, 3, 4]).values & ~t1 & (D.transactions_last_24h < 12).values & ~b, "-0.5")}
    for Kf, Kl in [(3, 8), (4, 12), (6, 16)]:
        c, info, _ = lca_correction(Kf, Kl)
        cc = c - np.median(c)
        print(f"Kf={Kf} Kl={Kl}: test fraud prior {info['test_prior']:.4f} (train 0.0177); fraud-class weights train {np.round(info['train_w'][:Kf] / info['train_w'][:Kf].sum(), 2)} test {np.round(info['test_w'][:Kf] / info['test_w'][:Kf].sum(), 2)}")
        print("   implied group shifts (centred) vs public optimum:", "  ".join(f"{k} {cc[m].mean():+.2f} [{pub}]" for k, (m, pub) in G.items()), flush=True)
        np.save(f"experiments/lca_corr_{Kf}_{Kl}.npy", c)
