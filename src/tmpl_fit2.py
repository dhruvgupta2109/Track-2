"""Joint fit of (s, q2, k, q1, q3) to ABSOLUTE public levels + differences, so prevalence scale cannot fake the level argument."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
import tmpl_fit as T
from sklearn.metrics import average_precision_score as ap
M = T.M; Rn = 80; rng = np.random.default_rng(3); U = rng.random((Rn, len(T.te))); SL = rng.random((Rn, len(T.te))) < 0.5
def expected(s, q2, k, q1=None, q3=None):
    p = np.clip(T.base * s, 0, 1); p[T.HV_OUT] = np.clip(p[T.HV_OUT] * k, 0, 1); p[T.T2] = q2
    if q1 is not None: p[M["T1"]] = q1
    if q3 is not None: p[M["T3"]] = q3
    out = np.zeros(len(T.S))
    for r in range(Rn):
        y = U[r] < p; m = SL[r]
        for i, f in enumerate(T.S.values()): out[i] += ap(y[m], f[m]) / Rn
    return out
rows = []
for s in (0.7, 1.0, 1.3, 1.7):
    for q2, k in [(0.0, 2.0), (0.0, 1.5), (0.005, 2.0), (0.02, 1.5), (0.05, 1.0), (0.05, 1.5), (0.08, 1.0)]:
        e = expected(s, q2, k); d = e - T.OBS; dc = (e - e.mean()) - (T.OBS - T.OBS.mean())
        rows.append(dict(s=s, q2=q2, k=k, rmse_abs=np.sqrt(np.mean(d ** 2)), rmse_centred=np.sqrt(np.mean(dc ** 2)), M123=e[-1], Z2=e[-4]))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv("experiments/tmpl_fit2.csv", index=False)
print(df.sort_values("rmse_abs").head(10).to_string()); print(df.sort_values("rmse_centred").head(10).to_string())
