import sys, warnings, hashlib; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from r9lab import make, v5_local, embed, class_dists, pooled_dists, EMB, TR
from evaluate import TE
import augment as G

def pool(gcol):
    def fn(A, B, y):
        Ea, Eb = embed(A, B, EMB); ga, gb = A[gcol].values, B[gcol].values
        fa, fb = pooled_dists(Ea, Eb, ga, gb, y, 1, 1); la, lb = pooled_dists(Ea, Eb, ga, gb, y, 0, 1)
        return pd.DataFrame({"dr": np.log1p(fa[:, 0]) - np.log1p(la[:, 0])}), pd.DataFrame({"dr": np.log1p(fb[:, 0]) - np.log1p(lb[:, 0])})
    return fn
def with_cash(A, B, y):
    for D in (A, B): D["cash"] = (D.merchant_category == "cash_transfer").astype(float)
    return v5_local(A, B, y, cols=EMB + ["cash"])
def multi_amt_age(A, B, y):
    a1, b1 = v5_local(A, B, y); a2, b2 = v5_local(A, B, y, cols=["log_amt", "amt_rel_cat", "log_age", "avg_spend"])
    return pd.DataFrame({"dr": a1.dr, "dr_aa": a2.dr}), pd.DataFrame({"dr": b1.dr, "dr_aa": b2.dr})
def emb_cat_hour(A, B, y):
    Ea, Eb = embed(A, B, EMB)
    for c in ["merchant_category", "transaction_channel", "country"]:
        lv = sorted(A[c].dropna().unique())
        Ea = np.c_[Ea, 0.5 * np.c_[tuple((A[c] == v).values for v in lv)]]; Eb = np.c_[Eb, 0.5 * np.c_[tuple((B[c] == v).values for v in lv)]]
    ha, hb = A.transaction_hour.fillna(12).values, B.transaction_hour.fillna(12).values
    Ea = np.c_[Ea, 0.5 * np.cos(2 * np.pi * ha / 24), 0.5 * np.sin(2 * np.pi * ha / 24)]; Eb = np.c_[Eb, 0.5 * np.cos(2 * np.pi * hb / 24), 0.5 * np.sin(2 * np.pi * hb / 24)]
    fa, fb = class_dists(Ea, Eb, y, 1, 1); la, lb = class_dists(Ea, Eb, y, 0, 1)
    return pd.DataFrame({"dr": np.log1p(fa[:, 0]) - np.log1p(la[:, 0])}), pd.DataFrame({"dr": np.log1p(fb[:, 0]) - np.log1p(lb[:, 0])})
def pseudo30(a, b):
    p0 = G.fit_predict(a, b); return G.fit_predict(a, b, None, b[p0 <= np.quantile(p0, 0.3)].assign(fraud=0, w=0.25))
def mp_smote(a, b, alpha=0.25):
    from exp_r11_aug2 import _fit
    rng = np.random.default_rng(len(a)); nf = int(a.fraud.sum()); syn = G.smote(a, 2000, rng); syn = syn.assign(w=alpha * nf / len(syn))
    return _fit(a, b, a.assign(w=np.where(a.fraud == 1, 1 - alpha, 1.0)), syn)

CANDS = {  # name: (generator, local d_std, d_hnW vs v5 on screen)
 "k01_knn_merchant_pool": (make(pool("merchant_category")), "+0.0011", "+0.0041"),
 "k02_knn_drop_amt_rel_avg": (make(lambda A, B, y: v5_local(A, B, y, cols=[c for c in EMB if c != "amt_rel_avg"])), "+0.0006", "+0.0012"),
 "k03_knn_add_cash": (make(with_cash), "+0.0002", "+0.0025"),
 "k04_knn_multimetric_amt_age": (make(multi_amt_age), "~+0.0010 (E1-based screen)", "~+0.0026"),
 "k05_knn_foreign_pool": (make(pool("foreign")), "+0.0004", "+0.0002"),
 "k06_knn_emb_cat0.5_hour0.5": (make(emb_cat_hour), "+0.0003", "-0.0004"),
 "k07_v5_C0.05": (make(lambda A, B, y: v5_local(A, B, y), C=0.05), "+0.0002", "-0.0017"),
 "k08_pseudo_legit30": (pseudo30, "+0.0003", "+0.0005"),
 "k09_masspres_smote0.25": (mp_smote, "+0.0003", "+0.0001"),
}
v5 = pd.read_csv("submissions/sub_v5_lr_knn.csv").prediction.values; r = lambda x: rankdata(x) / len(x)
rows = []
def save(name, p, ds, dh):
    p = np.clip(p, 0, 1); f = f"submissions/batch_small/{name}.csv"; pd.DataFrame({"id": TE.id, "prediction": p}).to_csv(f, index=False)
    rows.append((name, ds, dh, round(spearmanr(p, v5)[0], 4), len(set(np.argsort(-p)[:120]) & set(np.argsort(-v5)[:120])) / 120, hashlib.sha256(open(f, "rb").read()).hexdigest()[:10]))
    print(rows[-1], flush=True)
for n, (g, ds, dh) in CANDS.items(): save(n, g(TR, TE), ds, dh)
S = lambda f: pd.read_csv(f).prediction.values
save("k10_v5_rank95_subtype_device", 0.95 * r(v5) + 0.05 * r(S("submissions/audit/cand_subtype_device.csv")), "+0.0009", "+0.0020")
save("k11_v5_rank95_subtype_young", 0.95 * r(v5) + 0.05 * r(S("submissions/audit/cand_subtype_young.csv")), "+0.0005", "+0.0030")
save("k12_v5_prob90_catboost10", 0.9 * v5 + 0.1 * S("submissions/audit/cand_catboost_d4.csv"), "+0.0004", "-0.0001")
save("k13_v5_rank80_nca20", 0.8 * r(v5) + 0.2 * r(S("submissions/audit/deep_nca_v5design.csv")), "+0.0006", "+0.0003")
save("k14_v5_rank80_skmlp10_tabm10", 0.8 * r(v5) + 0.1 * r(S("submissions/audit/deep_skmlp_v5design.csv")) + 0.1 * r(S("submissions/audit/deep_tabm_v5design.csv")), "+0.0010", "+0.0022")
save("k15_v5_trans_q0.85", S("submissions/audit/v5_trans_q0.85.csv"), "+0.0002", "+0.0005")
pd.DataFrame(rows, columns=["file", "d_std_vs_v5", "d_hnW_vs_v5", "spearman_v5", "top1pct_overlap_v5", "sha256_10"]).to_csv("submissions/batch_small/INDEX.csv", index=False)
