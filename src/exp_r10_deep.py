import sys, warnings, time; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap
from deep import design_v5, train_net
from r9lab import TR, Y, SPLITS, W_HN, O5
from gpulock import gpu_lock
CFG = {"mlp_small10": ("mlp", dict(h=64, depth=2, p=0.3, plr=False, wd=1e-2), 10),
       "tabm32_reg": ("tabm", dict(k=32, h=64, depth=2, p=0.3, plr=False, wd=1e-2), 2),
       "nca_linear16": ("nca", dict(dim=16, plr=False, linear=True), 3),
       "nca_mlp32_noplr": ("nca", dict(dim=32, h=64, p=0.3, plr=False, wd=1e-2), 3),"mlp_big": ("mlp", dict(h=256, depth=3, p=0.2), 3), "tabm16": ("tabm", dict(k=16, h=128, depth=2), 2), "nca64": ("nca", dict(dim=64, h=128), 3)}
which = sys.argv[1:] or list(CFG)
reps = 2; folds = SPLITS[: 5 * reps]
rk = lambda x: np.vstack([rankdata(r) / len(r) for r in x])
f = lambda o, w=None: np.mean([ap(Y, o[r], sample_weight=w) for r in range(reps)])
with gpu_lock("track2-deep") as got:
    dev = "cuda" if got and torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
    D = [design_v5(TR.iloc[a], TR.iloc[b]) for a, b in folds]
    for nm in which:
        kind, kw, nseed = CFG[nm]; o = np.zeros((reps, len(Y))); t = time.time()
        for i, (a, b) in enumerate(folds):
            Xa, Xb, ya, nn_ = D[i]; o[i // 5, b] = np.mean([train_net(kind, Xa, ya, Xb, nn_, dev, seed=s, **kw) for s in range(nseed)], 0)
        np.save(f"experiments/oofscreen_{nm}.npy", o)
        print(f"{nm:10s} standalone std {f(o):.4f} hnW {f(o, W_HN):.4f}  ({time.time()-t:.0f}s)  | v5 std {f(O5[:reps]):.4f} hnW {f(O5[:reps], W_HN):.4f}", flush=True)
        for w in [0.1, 0.2, 0.3, 0.5]:
            b = (1 - w) * rk(O5[:reps]) + w * rk(o)
            print(f"     rank {1-w:.1f} v5 + {w} {nm}: d_std {f(b)-f(O5[:reps]):+.4f} d_hnW {f(b, W_HN)-f(O5[:reps], W_HN):+.4f}", flush=True)
