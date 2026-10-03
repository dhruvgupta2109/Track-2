import numpy as np, pandas as pd, itertools
from scipy.stats import rankdata
S = lambda f: pd.read_csv(f).prediction.values; r = lambda x: rankdata(x) / len(x)
SUR = "../Datathon Track 2 - suraj try/"
BASE = {"v5": S("submissions/sub_v5_lr_knn.csv"), "k01": S("submissions/batch_small/k01_knn_merchant_pool.csv"),
        "sur": S(SUR + "submission.csv"), "sur_c1": S(SUR + "extra_submissions/cand1_v1_plus_paysim_score.csv"),
        "sur_c3": S(SUR + "extra_submissions/cand3_blend_0.8v1_0.2lgbm.csv"), "mlp": S("submissions/audit/deep_skmlp_v5design.csv"),
        "v4": S("submissions/sub_v4_lr.csv"), "cb": S("submissions/audit/cand_catboost_d4.csv"), "surk": S("submissions/audit/sur_v1_knn.csv")}
R = {k: r(v) for k, v in BASE.items()}
def pool():
    out = {k: v for k, v in R.items()}
    ws = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for a, b in itertools.combinations(R, 2):
        for w in ws: out[f"{1-w:.1f}{a}+{w:.1f}{b}"] = (1 - w) * R[a] + w * R[b]
    for (a, b, c) in [("k01", "sur", "mlp"), ("v5", "sur", "mlp"), ("k01", "sur_c1", "mlp"), ("k01", "sur", "sur_c1"), ("v5", "k01", "sur")]:
        for wa, wb in [(0.5, 0.4), (0.5, 0.3), (0.4, 0.4), (0.6, 0.3), (0.4, 0.5), (0.3, 0.5)]:
            out[f"{wa}{a}+{wb}{b}+{1-wa-wb:.1f}{c}"] = wa * R[a] + wb * R[b] + (1 - wa - wb) * R[c]
    return out
