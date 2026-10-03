"""Build template-aware candidates on top of friend's Q30 (0.23288)."""
import sys, hashlib, numpy as np, pandas as pd
sys.path.insert(0, "src")
from templates import template_masks
te = pd.read_csv("test.csv"); q30 = pd.read_csv("../Track2_Q30_final_package_0.23288/predictions_Q30.csv")
assert (q30.id.values == te.id.values).all()
M = template_masks(te); T2 = M["T2"]
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9))); sg = lambda z: 1 / (1 + np.exp(-z))
a_ = te.account_age.values; b = (te.transactions_last_24h >= 16).values; ndold = (te.new_device == 1).values & (a_ > 365) & ~b
hvc = te.merchant_category.isin(["luxury", "cash_transfer", "electronics", "travel"]).values; amt = te.transaction_amount.values
hv = hvc & (amt >= 2000) & (amt <= 6000) & (a_ > 365) & ~b & ~ndold; mv = hvc & (amt >= 1000) & (amt < 2000) & (a_ > 365) & ~b & ~ndold
cur = np.where(hv, 1.0, np.where(mv, 0.5, 0.0)); OUT = hvc & (amt >= 1000) & (a_ > 365) & ~b & ~ndold & ~T2
def build(name, t2, out):
    z = lg(q30.prediction.values.copy()); z[T2] += -cur[T2] + t2; z[OUT] += out
    p = sg(z); assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    df = pd.DataFrame({"id": te.id, "prediction": p}); path = f"submissions/tmpl/{name}.csv"; df.to_csv(path, index=False)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:10]
    print(f"{path}: sha {sha}, rows changed vs Q30 {int((np.abs(p - q30.prediction.values) > 1e-12).sum())}, T2 rows {T2.sum()}, out rows {OUT.sum()}")
build("C1_Q30_T2box_unshift", 0.0, 0.0)
build("C7_Q30_T2box-1_hvout+0.5", -1.0, 0.5)
