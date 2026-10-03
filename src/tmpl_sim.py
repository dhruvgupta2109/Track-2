"""How fraud-rich must the injected high-value template be to explain Z2 -> Z3 -> K3 -> M123 public scores?
Label model: non-template rows ~ Bernoulli(s * p_M123(x)) (s scales prevalence), template rows ~ Bernoulli(q) (uniform inside template).
Expected AP over R random label draws x random 50% public slices; compared via paired differences to the observed public deltas."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, "src")
from features_prod import make_features
from constants import fit_constants

tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv"); D = make_features(te, fit_constants(tr))
ROOT = ".."
FILES = {"Z2": (f"{ROOT}/BEST_Z2_v5_youngburst-2_oldburst-0.5_ndold-0.75_0.21922", 0.21922), "Z3": ("submissions/shift/Z3_Z2_bigold+0.5.csv", 0.22215),
         "K3": (f"{ROOT}/Track2_K3_final_package_0.223/predictions_K3.csv", 0.223), "M123": (f"{ROOT}/Track2_M123_final_package_0.22523/predictions_M123.csv", 0.22523)}
import glob, os
def load(p):
    if os.path.isdir(p): p = [f for f in glob.glob(p + "/**/*.csv", recursive=True) if "pred" in os.path.basename(f).lower()][0]
    return pd.read_csv(p).prediction.values
S = {k: load(v[0]) for k, v in FILES.items()}; OBS = {k: v[1] for k, v in FILES.items()}
a_ = D.account_age.values; b = (D.transactions_last_24h >= 16).values; ndold = (D.new_device == 1).values & (a_ > 365) & ~b
hvc = D.merchant_category.isin(["luxury", "cash_transfer", "electronics", "travel"]).values; amt = D.transaction_amount.values
T = hvc & (amt >= 1500) & (amt <= 6000) & (a_ > 1100) & (D.new_device == 0).values & (D.transactions_last_24h <= 4).values & D.transaction_hour.between(8, 20).values
print("template rows", T.sum())
lg = lambda p: np.log(p / (1 - p)); sg = lambda z: 1 / (1 + np.exp(-z))


def expected_ap(q, s, files, R=150, seed=0, extra=None):
    rng = np.random.default_rng(seed); p = np.clip(S["M123"] * s, 0, 1); p[T] = q
    out = {k: [] for k in files}
    for r in range(R):
        y = rng.random(len(p)) < p; m = rng.random(len(p)) < 0.5
        for k in files: out[k].append(ap(y[m], files[k][m]))
    return {k: np.array(v) for k, v in out.items()}


if __name__ == "__main__":
    files = dict(S)
    for sh in (2.0, 3.0, 5.0):
        z = lg(np.clip(S["M123"], 1e-9, 1 - 1e-9)); z[T] += sh - 1.0; files[f"M123_T{sh:+.0f}"] = sg(z)
    rows = []
    for s in (0.6, 1.0):
        for q in (0.0, 0.02, 0.04, 0.07, 0.10, 0.15, 0.25):
            E = expected_ap(q, s, files)
            d = {k: (E[k] - E["Z2"]).mean() for k in files}
            rows.append(dict(s=s, q=q, Z2=E["Z2"].mean(), **{f"d_{k}": d[k] for k in files if k != "Z2"}))
            print(rows[-1], flush=True)
    print("observed deltas vs Z2:", {k: round(OBS[k] - OBS["Z2"], 4) for k in OBS})
    pd.DataFrame(rows).round(4).to_csv("experiments/tmpl_sim.csv", index=False)
