"""Sim value of each stacked piece on top of Q30 (truth grid incl. fitted T1/T2/nd_old results)."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, "src")
from templates import template_masks
te = pd.read_csv("test.csv"); M = template_masks(te); T1, T2, T3 = M["T1"], M["T2"], M["T3"]
q30 = pd.read_csv("../Track2_Q30_final_package_0.23288/predictions_Q30.csv").prediction.values
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9))); sg = lambda z: 1 / (1 + np.exp(-z))
b = (te.transactions_last_24h >= 16).values; NDB = (te.new_device == 1).values & (te.account_age.values > 365) & ~b & ~T1
def level(z, mask, q, alpha):
    z = z.copy(); zm = z[mask].mean(); z[mask] = lg(np.array([q]))[0] + alpha * (z[mask] - zm); return z
z0 = lg(q30)
C = {"Q30": z0}
for a in (0.0, 0.3, 0.6, 1.0): C[f"T2 .045 a{a}"] = level(z0, T2, 0.045, a)
C["ndbase -0.5"] = z0 + np.where(NDB, -0.5, 0.0)
C["T2 .045 a0.3 + ndbase -0.5"] = level(z0, T2, 0.045, 0.3) + np.where(NDB, -0.5, 0.0)
C["... + T3 .03"] = level(C["T2 .045 a0.3 + ndbase -0.5"], T3, 0.03, 0.3)
C["... + T3 .015"] = level(C["T2 .045 a0.3 + ndbase -0.5"], T3, 0.015, 0.3)
Rn = 200; rng = np.random.default_rng(21); U = rng.random((Rn, len(te))); SL = rng.random((Rn, len(te))) < 0.5
rows = []
for s in (1.0, 1.3):
    for q2, within in [(0.045, "uniform"), (0.045, "model"), (0.03, "uniform"), (0.06, "uniform")]:
        for knd in (0.5, 1.0):
            for q3 in (0.002, 0.03):
                p = np.clip(q30 * s, 0, 1); p[T1] = 0.002; p[T3] = q3; p[NDB] = np.clip(p[NDB] * knd, 0, 1)
                if within == "uniform": p[T2] = q2
                else: w = q30[T2] / q30[T2].mean(); p[T2] = np.clip(q2 * w, 0, 1)
                res = {n: [] for n in C}
                for r in range(Rn):
                    y = U[r] < p; m = SL[r]
                    for n, f in C.items(): res[n].append(ap(y[m], f[m]))
                mu = {n: np.mean(v) for n, v in res.items()}
                rows.append(dict(s=s, q2=q2, within=within, knd=knd, q3=q3, Q30=round(mu["Q30"], 4), **{n: round(mu[n] - mu["Q30"], 4) for n in C if n != "Q30"}))
                print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv("experiments/stack_sim.csv", index=False)
pd.set_option("display.width", 300); cols = [c for c in C if c != "Q30"]
print(df.to_string()); print("mean:", df[cols].mean().round(4).to_dict()); print("min:", df[cols].min().round(4).to_dict())
print("mean q3=.002:", df[df.q3 == 0.002][cols].mean().round(4).to_dict()); print("mean q3=.03:", df[df.q3 == 0.03][cols].mean().round(4).to_dict())
