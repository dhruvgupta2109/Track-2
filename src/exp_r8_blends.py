import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap
from evaluate import Y, W_HN, HOLD, boot_diff, prevalence_deltas
from candidates import REG, load_oof
o5 = load_oof("oof5_v5_prod")
rk = lambda x: np.vstack([rankdata(r) / len(r) for r in x])
lg = lambda p: np.log(np.clip(p, 1e-6, 1) / np.clip(1 - p, 1e-6, 1))
def views(o):
    return np.mean([ap(Y, o[i]) for i in range(5)]), np.mean([ap(Y, o[i], sample_weight=W_HN) for i in range(5)])
b5 = views(o5); rows = []
for n in ["lgb_mono", "catboost_d4", "histgb", "extratrees", "subtype_young", "subtype_device", "v4_burst_nd", "moe_newdevice", "v4_knn_full", "v5b_burstnight", "v1_lr_eng", "v3_lr"]:
    o = load_oof(REG[n][0])
    for w in [0.05, 0.1, 0.2, 0.3]:
        for kind, b in [("prob", (1 - w) * o5 + w * o), ("rank", (1 - w) * rk(o5) + w * rk(o)), ("logit", (1 - w) * lg(o5) + w * lg(o))]:
            s, h = views(b); rows.append((n, kind, w, s - b5[0], h - b5[1]))
df = pd.DataFrame(rows, columns=["partner", "kind", "w", "d_std", "d_hnW"])
best = df.sort_values("d_std", ascending=False).groupby("partner").head(1).sort_values("d_std", ascending=False)
print("best blend per partner (vs v5 std %.4f hnW %.4f):" % b5); print(best.round(4).to_string(index=False))
df.round(5).to_csv("experiments/r8_blends.csv", index=False)
# uncertainty + prevalence for the top few
for _, r in best.head(5).iterrows():
    o = load_oof(REG[r.partner][0]); w = r.w
    b = {"prob": (1 - w) * o5 + w * o, "rank": (1 - w) * rk(o5) + w * rk(o), "logit": (1 - w) * lg(o5) + w * lg(o)}[r.kind]
    (lo, _, hi), pw = boot_diff(b.mean(0), o5.mean(0)); (lo2, _, hi2), pw2 = boot_diff(b.mean(0), o5.mean(0), W_HN)
    print(f"{r.partner} {r.kind} w={w}: d_std90 {lo:+.4f}..{hi:+.4f} P>0 {pw:.2f} | d_hnW90 {lo2:+.4f}..{hi2:+.4f} P>0 {pw2:.2f} | prev",
          {k: round(v, 4) for k, v in prevalence_deltas(b, o5, prevs=(0.0075, 0.01, 0.0177), n=60).items()})
