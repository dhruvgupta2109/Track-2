"""Fresh-seed check: R2 base (0.8*(0.7 v5 + 0.3 poly) + 0.2 GALR) vs + w * degree-3 interaction LR (C=0.001); all models refit per fold."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, "src")
from build_v5 import fit_v5, predict_v5
from blendlab import fp_poly
from galr import fp as fp_galr
tr = pd.read_csv("train.csv"); y = tr.fraud.values
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
for seed in (11, 99, 314, 2718):
    o = {k: np.zeros(len(y)) for k in ("v5", "poly", "galr", "poly3")}
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y):
        A, B = tr.iloc[a], tr.iloc[b]
        o["v5"][b] = predict_v5(fit_v5(A), B); o["poly"][b] = fp_poly()(A, B); o["galr"][b] = fp_galr(0.05)(A, B); o["poly3"][b] = fp_poly(0.001, 3)(A, B)
    r2 = 0.8 * (0.7 * lg(o["v5"]) + 0.3 * lg(o["poly"])) + 0.2 * lg(o["galr"]); base = ap(y, r2)
    z3 = lg(o["poly3"]); z3 = (z3 - z3.mean()) / z3.std() * r2.std()
    print(f"seed {seed}: R2 base {base:.4f} | " + "  ".join(f"+poly3 w{w}: {ap(y, (1 - w) * r2 + w * z3) - base:+.4f}" for w in (0.05, 0.1, 0.15, 0.2)), flush=True)
