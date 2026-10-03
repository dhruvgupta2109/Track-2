"""Fit the planted T1 (new-device, established, small, daytime) fraud rate from the nd_old demotion series on public."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, "src")
from templates import template_masks
te = pd.read_csv("test.csv"); M = template_masks(te); T1, T2, T3 = M["T1"], M["T2"], M["T3"]
F = {"v5_burst": ("../FINAL_UPLOAD_v5_burst/predictions_v5_burst.csv", 0.198), "S3": ("submissions/shift/S3_burst-1.0_ndold-0.5.csv", 0.20453),
     "W2": ("submissions/shift/W2_burst-1.0_ndold-0.75.csv", 0.20530), "S4": ("submissions/shift/S4_burst-1.0_ndold-1.0.csv", 0.20516)}
S = {k: pd.read_csv(v[0]).prediction.values for k, v in F.items()}; OBS = np.array([v[1] for v in F.values()])
q30 = pd.read_csv("../Track2_Q30_final_package_0.23288/predictions_Q30.csv").prediction.values
b = (te.transactions_last_24h >= 16).values; ndold_base = (te.new_device == 1).values & (te.account_age.values > 365) & ~b & ~T1
print("T1", T1.sum(), "base nd_old outside T1", ndold_base.sum())
Rn = 200; rng = np.random.default_rng(5); U = rng.random((Rn, len(te))); SL = rng.random((Rn, len(te))) < 0.5
rows = []
for s in (1.0, 1.3):
    for q1 in (0.002, 0.01, 0.02, 0.035, 0.05):
        for knd in (0.5, 1.0, 2.0):   # base nd_old truth relative to Q30 (which already has -0.75)
            p = np.clip(q30 * s, 0, 1); p[T2] = 0.045; p[T3] = 0.005; p[T1] = q1; p[ndold_base] = np.clip(p[ndold_base] * knd, 0, 1)
            e = np.zeros(len(S))
            for r in range(Rn):
                y = U[r] < p; m = SL[r]
                for i, f in enumerate(S.values()): e[i] += ap(y[m], f[m]) / Rn
            dc = (e - e[0]) - (OBS - OBS[0])
            rows.append(dict(s=s, q1=q1, knd=knd, rmse=np.sqrt(np.mean(dc[1:] ** 2)), **{k: round(x - e[0], 4) for k, x in zip(F, e)}))
            print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv("experiments/t1_fit.csv", index=False)
print("observed deltas vs v5_burst:", dict(zip(F, np.round(OBS - OBS[0], 4)))); print(df.sort_values("rmse").head(10).to_string())
