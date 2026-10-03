"""Greedy forward selection of extra blend members on top of the R2 base (0.8*(0.7 v5 + 0.3 poly) + 0.2 GALR), logit space.
Select on OOF repeats 0-2, confirm on held-out repeats 3-4 (same data, different fold assignments)."""
import sys, glob, numpy as np
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, "src")
from evaluate import Y
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
O5 = np.load("experiments/oof5_v5_prod.npy"); PO = np.load("experiments/oof5_bl_poly.npy"); GA = np.load("experiments/oof5_galr_C0.05_int.npy")
BASE = np.array([0.8 * (0.7 * lg(O5[r]) + 0.3 * lg(PO[r])) + 0.2 * lg(GA[r]) for r in range(5)])
C = {}
for f in glob.glob("experiments/oof5_*.npy"):
    o = np.load(f)
    if o.shape != (5, 20000) or np.isnan(o).any() or "v5_prod" in f or "bl_poly.npy" in f or "galr_C0.05_int" in f: continue
    z = np.array([lg(o[r]) for r in range(5)]); s = z.std(1, keepdims=True); C[f.split("oof5_")[1][:-4]] = (z - z.mean(1, keepdims=True)) / s * BASE.std(1, keepdims=True)
print(len(C), "candidates")
sel = lambda Z, reps: np.mean([ap(Y, Z[r]) for r in reps])
cur = BASE.copy(); chosen = []
for step in range(4):
    best = None
    for k, z in C.items():
        if k in [c for c, _ in chosen]: continue
        for w in (0.05, 0.1, 0.2):
            g = sel((1 - w) * cur + w * z, (0, 1, 2)) - sel(cur, (0, 1, 2))
            if best is None or g > best[0]: best = (g, k, w)
    g, k, w = best; nz = (1 - w) * cur + w * C[k]
    hold = sel(nz, (3, 4)) - sel(cur, (3, 4)); per = [round(ap(Y, nz[r]) - ap(Y, cur[r]), 4) for r in range(5)]
    print(f"step {step}: + {k} w={w}: select-reps {g:+.4f} | held-out reps {hold:+.4f} | per-rep {per}", flush=True)
    if hold <= 0: print("stop: no held-out gain"); break
    cur = nz; chosen.append((k, w))
print("base all reps %.4f -> %.4f" % (sel(BASE, range(5)), sel(cur, range(5))), chosen)
