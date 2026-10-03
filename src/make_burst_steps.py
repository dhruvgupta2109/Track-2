"""Coordinate steps on the burst-by-country split (B1 = R2op + MY/ID/SG bursts -0.25, other-country bursts +0.25 -> public 0.24016)."""
import sys, hashlib, numpy as np, pandas as pd
te = pd.read_csv("test.csv"); r2 = pd.read_csv("submissions/tmpl/R2op_package_output.csv"); assert (r2.id.values == te.id.values).all()
lg = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12))); sg = lambda z: 1 / (1 + np.exp(-z))
z0 = lg(r2.prediction.values); burst = (te.transactions_last_24h >= 16).values
pu = te.country.isin(["MY", "ID", "SG"]).values & burst; oth = te.country.notna().values & ~te.country.isin(["MY", "ID", "SG"]).values & burst
def build(name, d_pu, d_oth):
    z = z0.copy(); z[pu] += d_pu; z[oth] += d_oth; p = sg(z); assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    path = f"submissions/probe/{name}.csv"; pd.DataFrame({"id": te.id, "prediction": p}).to_csv(path, index=False)
    print(f"{path}: sha {hashlib.sha256(open(path, 'rb').read()).hexdigest()[:10]}  (MY/ID/SG bursts {d_pu:+}, other-country bursts {d_oth:+}; rows {pu.sum()}/{oth.sum()})")
    return p
b1 = build("B1check_R2op_burst_MYIDSG-0.25_other+0.25", -0.25, 0.25)
old = pd.read_csv("submissions/probe/B1_R2op_burst_MYIDSG-0.25_other+0.25.csv").prediction.values
print("B1 rebuild identical to scored B1:", np.abs(b1 - old).max() < 1e-15)
for name, a, o in [("B2a_burst_MYIDSG-0.25_other+0.5", -0.25, 0.5), ("B2b_burst_MYIDSG-0.5_other+0.25", -0.5, 0.25)]: build(name, a, o)
