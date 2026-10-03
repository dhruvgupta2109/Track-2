"""Population-level probabilities for planted templates: template rows get a constant level (random labels inside a uniform box).
Truth grid: base = Q30 probs * s; T2 uniform q2; T1 uniform q1; T3 uniform q3. Candidates: Q30 with templates set to levels."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, "src")
from templates import template_masks
te = pd.read_csv("test.csv"); M = template_masks(te); T1, T2, T3 = M["T1"], M["T2"], M["T3"]
q30 = pd.read_csv("../Track2_Q30_final_package_0.23288/predictions_Q30.csv").prediction.values
c1 = pd.read_csv("submissions/tmpl/C1_Q30_T2box_unshift.csv").prediction.values
c7 = pd.read_csv("submissions/tmpl/C7_Q30_T2box-1_hvout+0.5.csv").prediction.values
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9))); sg = lambda z: 1 / (1 + np.exp(-z))
def level(base, mask, q, alpha=0.05):
    z = lg(base.copy()); zm = z[mask].mean(); z[mask] = lg(np.array([q]))[0] + alpha * (z[mask] - zm); return sg(z)
def shift(base, mask, d):
    z = lg(base.copy()); z[mask] += d; return sg(z)
print("Q30 mean prob: T1 %.4f T2 %.4f T3 %.4f; ranks median T1 %d T2 %d T3 %d" % (q30[T1].mean(), q30[T2].mean(), q30[T3].mean(),
      *[np.median(pd.Series(q30).rank(ascending=False).values[m]) for m in (T1, T2, T3)]))
C = {"Q30": q30, "C1": c1, "C7": c7}
for q in (0.03, 0.045, 0.06): C[f"T2 lvl{q}"] = level(q30, T2, q)
C["T2 +0.5 vs C1 (shift)"] = shift(c1, T2 & ~(c1 == q30), 0.5)
C["T2 lvl.045 + T1 lvl.03"] = level(C["T2 lvl0.045"], T1, 0.03); C["T2 lvl.045 + T3 lvl.03"] = level(C["T2 lvl0.045"], T3, 0.03)
C["T2 lvl.045 + T1,T3 lvl.003"] = level(level(C["T2 lvl0.045"], T1, 0.003), T3, 0.003)
OBS = {"Q30": 0.23288, "C1": 0.23277, "C7": 0.22841}
Rn = 200; rng = np.random.default_rng(11); U = rng.random((Rn, len(te))); SL = rng.random((Rn, len(te))) < 0.5
rows = []
for s in (1.0, 1.3):
    for q2 in (0.03, 0.045, 0.06):
        for q1, q3 in [(0.003, 0.003), (0.03, 0.003), (0.003, 0.03), (0.05, 0.05)]:
            p = np.clip(q30 * s, 0, 1); p[T2] = q2; p[T1] = q1; p[T3] = q3
            res = {n: [] for n in C}
            for r in range(Rn):
                y = U[r] < p; m = SL[r]
                for n, f in C.items(): res[n].append(ap(y[m], f[m]))
            mu = {n: np.mean(v) for n, v in res.items()}
            fit = np.sqrt(np.mean([((mu[k] - mu["Q30"]) - (OBS[k] - OBS["Q30"])) ** 2 for k in ("C1", "C7")]))
            row = dict(s=s, q2=q2, q1=q1, q3=q3, Q30=round(mu["Q30"], 4), fitC1C7=round(fit, 4), **{n: round(mu[n] - mu["Q30"], 4) for n in C if n != "Q30"})
            rows.append(row); print(row, flush=True)
df = pd.DataFrame(rows); df.to_csv("experiments/tmpl_level.csv", index=False)
pd.set_option("display.width", 300); print(df.sort_values("fitC1C7").to_string())
