"""Q30-based candidates scored under a grid of label hypotheses (T2 fraud rate q2, out-of-box high-value multiplier k, T3 rate q3, scale s)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
import tmpl_fit as T
from sklearn.metrics import average_precision_score as ap
q30 = pd.read_csv("../Track2_Q30_final_package_0.23288/predictions_Q30.csv").prediction.values
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9))); sg = lambda z: 1 / (1 + np.exp(-z))
te = T.te; a_ = te.account_age.values; b = (te.transactions_last_24h >= 16).values; ndold = (te.new_device == 1).values & (a_ > 365) & ~b
hvc = te.merchant_category.isin(["luxury", "cash_transfer", "electronics", "travel"]).values; amt = te.transaction_amount.values
hv = hvc & (amt >= 2000) & (amt <= 6000) & (a_ > 365) & ~b & ~ndold; mv = hvc & (amt >= 1000) & (amt < 2000) & (a_ > 365) & ~b & ~ndold
T2, T3, OUT = T.T2, T.M["T3"], T.HV_OUT
cur = np.where(hv, 1.0, np.where(mv, 0.5, 0.0))
def cand(t2=None, t3=0.0, out=0.0):
    z = lg(q30)
    if t2 is not None: z[T2] += -cur[T2] + t2
    z[T3] += t3; z[OUT] += out; return sg(z)
C = {"Q30": q30, "C1 T2->0": cand(0.0), "C2 T2->-1": cand(-1.0), "C3 T2->+0.5": cand(0.5), "C4 T2->0,T3-1": cand(0.0, -1.0),
     "C5 T2->-2,out+0.5": cand(-2.0, 0.0, 0.5), "C6 T3-1 only": cand(None, -1.0)}
Rn = 150; rng = np.random.default_rng(7); U = rng.random((Rn, len(te))); SL = rng.random((Rn, len(te))) < 0.5
rows = []
for s in (1.0, 1.2):
    for q2, k in [(0.004, 1.0), (0.004, 2.0), (0.02, 1.5), (0.035, 1.2), (0.05, 1.0)]:
        for q3 in (0.002, 0.01):
            p = np.clip(T.base * s, 0, 1); p[OUT] = np.clip(p[OUT] * k, 0, 1); p[T2] = q2; p[T3] = q3
            res = {n: [] for n in C}
            for r in range(Rn):
                y = U[r] < p; m = SL[r]
                for n, f in C.items(): res[n].append(ap(y[m], f[m]))
            q = np.array(res["Q30"]); row = dict(s=s, q2=q2, k=k, q3=q3, Q30=q.mean())
            row.update({n: np.mean(v) - q.mean() for n, v in res.items() if n != "Q30"}); rows.append(row)
            print({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in row.items()}, flush=True)
df = pd.DataFrame(rows); df.to_csv("experiments/tmpl_decide2.csv", index=False)
print(df.round(4).to_string()); print("min over hypotheses:", df[[c for c in C if c != "Q30"]].min().round(4).to_dict())
print("mean over hypotheses:", df[[c for c in C if c != "Q30"]].mean().round(4).to_dict())
