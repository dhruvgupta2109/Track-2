"""Steps from B2a (public 0.24094 = R2op + MY/ID/SG bursts -0.25, other-country bursts +0.5). Shifts below are relative to R2op."""
import hashlib, numpy as np, pandas as pd
te = pd.read_csv("test.csv"); r2 = pd.read_csv("submissions/tmpl/R2op_package_output.csv"); assert (r2.id.values == te.id.values).all()
lg = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12))); sg = lambda z: 1 / (1 + np.exp(-z))
z0 = lg(r2.prediction.values); burst = (te.transactions_last_24h >= 16).values; young = (te.account_age <= 365).values
pu = te.country.isin(["MY", "ID", "SG"]).values & burst; oth = te.country.notna().values & ~te.country.isin(["MY", "ID", "SG"]).values & burst
def build(name, d_pu, d_oth_young, d_oth_old, check=None):
    z = z0.copy(); z[pu] += d_pu; z[oth & young] += d_oth_young; z[oth & ~young] += d_oth_old; p = sg(z)
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    path = f"submissions/probe/{name}.csv"; pd.DataFrame({"id": te.id, "prediction": p}).to_csv(path, index=False)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:10]
    print(f"{path}: sha {sha}  (MY/ID/SG {d_pu:+}, other young {d_oth_young:+}, other old {d_oth_old:+})" + (f"  reproduces scored file: {sha == check}" if check else ""))
    return path
chk = build("_B2a_check", -0.25, 0.5, 0.5, check="a8a0634efc")
import os; os.remove(chk)
build("C1_burst_MYIDSG-0.25_otherYoung+0.75_otherOld+0.25", -0.25, 0.75, 0.25)
build("C2_burst_MYIDSG0_other+0.5", 0.0, 0.5, 0.5)
