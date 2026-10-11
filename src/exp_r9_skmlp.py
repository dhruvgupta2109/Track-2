import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score as ap
from exp_r9_nn import design, train_mlp
from r9lab import *
from evaluate import TE
from gpulock import gpu_lock
import torch
def skmlp(Xa, ya, Xb, seeds=5):
    return np.mean([MLPClassifier((64, 32), alpha=1e-2, learning_rate_init=1e-3, max_iter=200, early_stopping=True, validation_fraction=0.125,
                                  n_iter_no_change=10, random_state=s).fit(Xa, ya).predict_proba(Xb)[:, 1] for s in range(seeds)], 0)
if __name__ == "__main__":
  reps = 2; o = np.zeros((reps, len(Y)))
  for i, (ia, ib) in enumerate(SPLITS[: 5 * reps]):
      Xa, Xb, ya, *_ = design(TR.iloc[ia], TR.iloc[ib]); o[i // 5, ib] = skmlp(Xa, ya, Xb)
  np.save("experiments/oofscreen_skmlp.npy", o)
  o6 = np.load("experiments/oof5_v6_E1.npy")[:2]; rk = lambda x: np.vstack([rankdata(r) / len(r) for r in x])
  f = lambda o: (np.mean([ap(Y, o[r]) for r in range(2)]), np.mean([ap(Y, o[r], sample_weight=W_HN) for r in range(2)]))
  print("sklearn MLP standalone", np.round(f(o), 4)); b6 = f(o6)
  for w in [0.1, 0.2]:
      s = f((1 - w) * rk(o6) + w * rk(o)); print(f"v6 rank + {w} skMLP: d_std {s[0]-b6[0]:+.4f} d_hnW {s[1]-b6[1]:+.4f}")
  # test predictions (full train): torch MLP and sklearn MLP on E1 design
  Xa, Xb, ya, *_ = design(TR, TE)
  with gpu_lock("track2-nn") as got:
      dev = "cuda" if got and torch.cuda.is_available() else "cpu"
      pt = np.mean([train_mlp(Xa, ya, Xb, dev, s) for s in range(5)], 0)
  ps = skmlp(Xa, ya, Xb)
  v6 = pd.read_csv("submissions/audit/v6_E1_knn_metric.csv").prediction.values; v5 = pd.read_csv("submissions/sub_v5_lr_knn.csv").prediction.values
  r = lambda x: rankdata(x) / len(x)
  for nm, p in [("v6_rank80_torchmlp20", 0.8 * r(v6) + 0.2 * r(pt)), ("v6_rank80_skmlp20", 0.8 * r(v6) + 0.2 * r(ps))]:
      pd.DataFrame({"id": TE.id, "prediction": p}).to_csv(f"submissions/audit/{nm}.csv", index=False)
      print(nm, "spearman v5", round(spearmanr(p, v5)[0], 4), "v6", round(spearmanr(p, v6)[0], 4))
  pd.DataFrame({"id": TE.id, "prediction": pt}).to_csv("submissions/audit/mlp_torch_E1.csv", index=False)
  print("torch MLP vs v5 spearman", round(spearmanr(pt, v5)[0], 4), "sk vs torch", round(spearmanr(ps, pt)[0], 4))
