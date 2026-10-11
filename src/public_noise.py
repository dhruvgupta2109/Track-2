"""How much of each public gain could be chance? Labels ~ Bernoulli(s * p) with p = C2's pre-operating-point probability (s matched to the
public level), random 50% public halves; report the spread (sd) of AP differences between our files. Also the optimism of picking the best
of several near-equal files (winner's curse)."""
import sys, numpy as np, pandas as pd, joblib, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, "src")
from r2_runtime import predict_r2
te = pd.read_csv("test.csv")
lg = lambda x: np.log(np.clip(x, 1e-12, 1 - 1e-12) / (1 - np.clip(x, 1e-12, 1 - 1e-12))); sg = lambda z: 1 / (1 + np.exp(-z))
F = {"R2op": "submissions/tmpl/R2op_package_output.csv", "B1": "submissions/probe/B1_R2op_burst_MYIDSG-0.25_other+0.25.csv",
     "B2a": "submissions/probe/B2a_burst_MYIDSG-0.25_other+0.5.csv", "C2": "submissions/probe/C2_burst_MYIDSG0_other+0.5.csv",
     "D1": "submissions/probe/D1_burst_MYIDSG0_otherYoung+0.5_otherOld+1.0.csv", "Q30": "../Track2_Q30_final_package_0.23288/predictions_Q30.csv"}
OBS = {"R2op": 0.23434, "B1": 0.24016, "B2a": 0.24094, "C2": 0.24174, "D1": 0.23604, "Q30": 0.23288}
S = {k: pd.read_csv(v).prediction.values for k, v in F.items()}
p = sg(lg(S["C2"]) - 2.188389)                     # C2 ranking probability before the operating-point shift
rng = np.random.default_rng(0); R = 400
for s in (1.0, 1.3):
    q = np.clip(s * p, 0, 1); res = {k: [] for k in S}
    for r in range(R):
        y = rng.random(len(q)) < q; m = rng.random(len(q)) < 0.5
        for k, v in S.items(): res[k].append(ap(y[m], v[m]))
    A = {k: np.array(v) for k, v in res.items()}
    print(f"s={s}: mean public AP of C2 under this label model {A['C2'].mean():.4f} (observed 0.24174)")
    for a, b in [("C2", "R2op"), ("C2", "Q30"), ("C2", "B2a"), ("C2", "B1"), ("C2", "D1")]:
        d = A[a] - A[b]
        print(f"   {a}-{b}: chance spread sd {d.std():.4f} | observed {OBS[a] - OBS[b]:+.4f} = {(OBS[a] - OBS[b]) / d.std():+.1f} sd")
    # winner's curse: best of the 6 burst-family files when their true quality were equal (use the AP noise of near-identical rankings)
    fam = np.vstack([A[k] for k in ("R2op", "B1", "B2a", "C2", "D1")])
    dev = fam - fam.mean(0, keepdims=True)
    print(f"   optimism of picking the max of 5 files of equal true quality: {dev.max(0).mean():.4f}")
