import sys, warnings, hashlib; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd, torch
from scipy.stats import rankdata, spearmanr
from deep import design_v5, train_net
from exp_r10_blend import skmlp
from evaluate import TR, TE
from gpulock import gpu_lock
Xa, Xb, ya, nn_ = design_v5(TR, TE)
P = {"skmlp": skmlp(Xa, ya, Xb)}
with gpu_lock("track2-deep") as got:
    dev = "cuda" if got and torch.cuda.is_available() else "cpu"
    P["mlp"] = np.mean([train_net("mlp", Xa, ya, Xb, nn_, dev, seed=s, h=64, depth=2, p=0.3, plr=False, wd=1e-2) for s in range(10)], 0)
    P["tabm"] = np.mean([train_net("tabm", Xa, ya, Xb, nn_, dev, seed=s, k=32, h=64, depth=2, p=0.3, plr=False, wd=1e-2) for s in range(2)], 0)
    P["nca"] = np.mean([train_net("nca", Xa, ya, Xb, nn_, dev, seed=s, dim=32, h=64, p=0.3, plr=False, wd=1e-2) for s in range(3)], 0)
v5 = pd.read_csv("submissions/sub_v5_lr_knn.csv").prediction.values; r = lambda x: rankdata(x) / len(x)
out = {"v5_rank85_skmlp15": 0.85 * r(v5) + 0.15 * r(P["skmlp"]),
       "v5_rank80_deep4_20": 0.8 * r(v5) + 0.2 * np.mean([r(v) for v in P.values()], 0)}
for k, v in P.items(): pd.DataFrame({"id": TE.id, "prediction": v}).to_csv(f"submissions/audit/deep_{k}_v5design.csv", index=False)
for n, p in out.items():
    f = f"submissions/audit/{n}.csv"; pd.DataFrame({"id": TE.id, "prediction": p}).to_csv(f, index=False)
    print(n, "sha256", hashlib.sha256(open(f, "rb").read()).hexdigest()[:10], "spearman v5", round(spearmanr(p, v5)[0], 4),
          "top1% overlap", len(set(np.argsort(-p)[:120]) & set(np.argsort(-v5)[:120])) / 120)
