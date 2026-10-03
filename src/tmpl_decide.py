"""Expected public AP of Q30-based candidates under the competing label hypotheses (fit levels + differences)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
import tmpl_fit as T
from sklearn.metrics import average_precision_score as ap
q30 = pd.read_csv("../Track2_Q30_final_package_0.23288/predictions_Q30.csv").prediction.values
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9))); sg = lambda z: 1 / (1 + np.exp(-z))
te = T.te; a_ = te.account_age.values; b = (te.transactions_last_24h >= 16).values; ndold = (te.new_device == 1).values & (a_ > 365) & ~b
hvc = te.merchant_category.isin(["luxury", "cash_transfer", "electronics", "travel"]).values; amt = te.transaction_amount.values
hv = hvc & (amt >= 2000) & (amt <= 6000) & (a_ > 365) & ~b & ~ndold; mv = hvc & (amt >= 1000) & (amt < 2000) & (a_ > 365) & ~b & ~ndold
T2 = T.T2; OUT = T.HV_OUT
def cand(t2_shift, out_extra=0.0):
    z = lg(q30); cur = np.where(hv, 1.0, np.where(mv, 0.5, 0.0))
    z[T2] += -cur[T2] + t2_shift; z[OUT] += out_extra; return sg(z)
C = {"Q30": q30, "T2 0": cand(0.0), "T2 -1": cand(-1.0), "T2 -2": cand(-2.0), "T2 -3": cand(-3.0), "T2 -2, out +0.5": cand(-2.0, 0.5), "T2 -1, out +0.5": cand(-1.0, 0.5), "T2 +1 (=Q30), out +0.5": cand(1.0 * 0 + 0.0, 0.0) * 0 + cand(None if False else 0.0, 0.0) * 0 + sg(lg(q30) + 0.5 * OUT)}
print({k: int((v != q30).sum()) for k, v in C.items()})
Rn = 200; rng = np.random.default_rng(1); U = rng.random((Rn, len(te))); SL = rng.random((Rn, len(te))) < 0.5
for (q2, k, s) in [(0.0, 2.0, 1.0), (0.005, 2.0, 1.0), (0.05, 1.0, 1.0), (0.02, 1.5, 1.0), (0.0, 1.0, 1.0)]:
    p = np.clip(T.base * s, 0, 1); p[OUT] = np.clip(p[OUT] * k, 0, 1); p[T2] = q2
    res = {n: [] for n in C}
    for r in range(Rn):
        y = U[r] < p; m = SL[r]
        for n, f in C.items(): res[n].append(ap(y[m], f[m]))
    q = np.array(res["Q30"])
    print(f"q2={q2} k={k}: Q30 {q.mean():.4f} | " + "  ".join(f"{n}: {np.mean(v) - q.mean():+.4f} (P>Q30 {np.mean(np.array(v) > q):.2f})" for n, v in res.items() if n != "Q30"), flush=True)
