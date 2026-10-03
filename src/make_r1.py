"""R1 = Q30 + planted-T2 rows set to a near-constant level (+ optional operating-point shift so 0.5 = top ~2.5%)."""
import sys, hashlib, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, "src")
from templates import template_masks
te = pd.read_csv("test.csv"); q30 = pd.read_csv("../Track2_Q30_final_package_0.23288/predictions_Q30.csv"); c1 = pd.read_csv("submissions/tmpl/C1_Q30_T2box_unshift.csv")
assert (q30.id.values == te.id.values).all()
T2 = template_masks(te)["T2"]
lg = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12))); sg = lambda z: 1 / (1 + np.exp(-z))
z30, zc1 = lg(q30.prediction.values), lg(c1.prediction.values)
print("T2 mean logit: Q30 %.2f (p %.3f)  C1 %.2f (p %.3f)  -> public equal, midpoint p %.3f" % (z30[T2].mean(), sg(z30[T2].mean()), zc1[T2].mean(), sg(zc1[T2].mean()), sg((z30[T2].mean() + zc1[T2].mean()) / 2)))
LEVEL, ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04, 0.3
z = z30.copy(); z[T2] = lg(np.array([LEVEL]))[0] + ALPHA * (z30[T2] - z30[T2].mean())
p = sg(z)
thr = np.quantile(p, 1 - 0.025); C_OP = -lg(np.array([thr]))[0]
p_op = sg(lg(p) + C_OP)
print(f"operating-point shift C = {C_OP:.3f} (score {thr:.4f} -> 0.5); flagged >=0.5: {(p_op >= 0.5).sum()} rows; rank corr p vs p_op {spearmanr(p, p_op)[0]:.6f}; ties before/after {len(p) - len(np.unique(p))}/{len(p) - len(np.unique(p_op))}")
for name, v in [(f"R1_Q30_T2level{LEVEL}", p), (f"R1op_Q30_T2level{LEVEL}_op2.5pct", p_op)]:
    path = f"submissions/tmpl/{name}.csv"; pd.DataFrame({"id": te.id, "prediction": v}).to_csv(path, index=False)
    print(path, "sha", hashlib.sha256(open(path, "rb").read()).hexdigest()[:10], "rows changed vs Q30", int((np.abs(v - q30.prediction.values) > 1e-12).sum()))
