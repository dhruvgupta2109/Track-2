"""One-change probes on top of R2op (public 0.23434). Each CSV changes exactly one group so its public delta is attributable."""
import sys, hashlib, numpy as np, pandas as pd
sys.path.insert(0, "src")
from templates import template_masks
te = pd.read_csv("test.csv"); r2 = pd.read_csv("submissions/tmpl/R2op_package_output.csv"); assert (r2.id.values == te.id.values).all()
lg = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12))); sg = lambda z: 1 / (1 + np.exp(-z))
z0 = lg(r2.prediction.values); M = template_masks(te)
burst = (te.transactions_last_24h >= 16).values
pu = te.country.isin(["MY", "ID", "SG"]).values; oth = te.country.notna().values & ~pu
def save(name, z, changed):
    p = sg(z); assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    path = f"submissions/probe/{name}.csv"; pd.DataFrame({"id": te.id, "prediction": p}).to_csv(path, index=False)
    rk0 = pd.Series(z0).rank(ascending=False).values; rk1 = pd.Series(z).rank(ascending=False).values
    print(f"{path}: sha {hashlib.sha256(open(path, 'rb').read()).hexdigest()[:10]}, rows changed {changed.sum()}, "
          f"changed rows in R2op top-300 {int((changed & (rk0 <= 300)).sum())}, flagged>=0.5 {(p >= 0.5).sum()}")
SPLIT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
z = z0.copy(); z[burst & pu] -= SPLIT; z[burst & oth] += SPLIT
save(f"B1_R2op_burst_MYIDSG-{SPLIT}_other+{SPLIT}", z, burst & (pu | oth))
z = z0.copy(); z[M["T3"]] += 0.7
save("T3a_R2op_T3box+0.7", z, M["T3"])
