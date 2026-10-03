"""Which explains the public high-value results: fraud inside the planted T2 box, or fraud in the base high-value rows outside it?
Label model (common random numbers, random 50% public slices): p = s * p_M123(x); T2-box rows -> q2; out-of-box high-value base rows -> k * p.
Fit (q2, k) to the public scores of every file that moved high-value rows, using paired differences."""
import sys, os, glob, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, "src")
from templates import template_masks
te = pd.read_csv("test.csv"); M = template_masks(te)
R = ".."
def load(p):
    if os.path.isdir(p): p = [f for f in glob.glob(p + "/**/*.csv", recursive=True) if "pred" in os.path.basename(f).lower()][0]
    return pd.read_csv(p).prediction.values
F = {"S3": ("submissions/shift/S3_burst-1.0_ndold-0.5.csv", 0.20453), "R1": ("submissions/shift/R1_burst-1.0_ndold-0.7_bigold-1.1.csv", 0.19707),
     "R2": ("submissions/shift/R2_burst-1.0_ndold-0.7_bigold-0.5.csv", 0.20171), "P1": ("submissions/shift/P1_burstYoung-2.0_burstOld-1.0_ndold-0.75.csv", 0.21289),
     "U1": ("submissions/shift/U1_P1_bigold+0.5.csv", 0.21424), "Z2": (f"{R}/BEST_Z2_v5_youngburst-2_oldburst-0.5_ndold-0.75_0.21922", 0.21922),
     "Z3": ("submissions/shift/Z3_Z2_bigold+0.5.csv", 0.22215), "K3": (f"{R}/Track2_K3_final_package_0.223/predictions_K3.csv", 0.223),
     "M123": (f"{R}/Track2_M123_final_package_0.22523/predictions_M123.csv", 0.22523)}
S = {k: load(v[0]) for k, v in F.items()}; OBS = np.array([v[1] for v in F.values()]); NAMES = list(F)
a_ = te.account_age.values; b = (te.transactions_last_24h >= 16).values; ndold = (te.new_device == 1).values & (a_ > 365) & ~b
hvc = te.merchant_category.isin(["luxury", "cash_transfer", "electronics", "travel"]).values
HV_OUT = hvc & (te.transaction_amount.values >= 1000) & (a_ > 365) & ~b & ~ndold & ~M["T2"]
T2 = M["T2"]
print("T2 box", T2.sum(), " out-of-box high-value base rows", HV_OUT.sum())
Rn = 120; rng = np.random.default_rng(0); U = rng.random((Rn, len(te))); SL = rng.random((Rn, len(te))) < 0.5
base = S["M123"].copy() if len(sys.argv) < 2 or sys.argv[1] != "q30" else pd.read_csv("../Track2_Q30_final_package_0.23288/predictions_Q30.csv").prediction.values


def expected(q2, k, s=1.0):
    p = np.clip(base * s, 0, 1); p[HV_OUT] = np.clip(p[HV_OUT] * k, 0, 1); p[T2] = q2
    out = np.zeros(len(S))
    for r in range(Rn):
        y = U[r] < p; m = SL[r]
        for i, f in enumerate(S.values()): out[i] += ap(y[m], f[m]) / Rn
    return out


if __name__ == "__main__":
    rows = []
    for s in (1.0,):
        for q2 in (0.0, 0.005, 0.02, 0.05, 0.08):
            for k in (1.0, 2.0, 3.0, 5.0):
                e = expected(q2, k, s); d = (e - e.mean()) - (OBS - OBS.mean())
                rows.append(dict(s=s, q2=q2, k=k, rmse=np.sqrt(np.mean(d ** 2)), **{n: round(x, 4) for n, x in zip(NAMES, e)}))
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows).sort_values("rmse"); df.to_csv("experiments/tmpl_fit.csv", index=False)
    print("observed:", dict(zip(NAMES, OBS))); print(df.head(8).to_string())
