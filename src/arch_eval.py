"""Common harness for new architectures: standalone AP and value as a blend member of the R2 base, on evaluate.SPLITS OOF (5 repeats).
R2 base = 0.8*(0.7 v5 + 0.3 interaction LR) + 0.2 GALR (logits) - the base behind public 0.23434.
Blend weight is chosen on repeats 0-2 and checked on held-out repeats 3-4 (same rows, different fold assignments)."""
import sys, numpy as np
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, "src")
from evaluate import Y
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
O5 = np.load("experiments/oof5_v5_prod.npy"); PO = np.load("experiments/oof5_bl_poly.npy"); GA = np.load("experiments/oof5_galr_C0.05_int.npy")
R2 = np.array([0.8 * (0.7 * lg(O5[r]) + 0.3 * lg(PO[r])) + 0.2 * lg(GA[r]) for r in range(5)])
WS = (0.05, 0.1, 0.2, 0.3, 0.5)


def evaluate(oof, name, base=R2, quiet=False):
    reps = [r for r in range(len(oof)) if not np.isnan(oof[r]).any()]
    own = np.array([ap(Y, oof[r]) for r in reps]); b = np.array([ap(Y, base[r]) for r in reps]); v5 = np.array([ap(Y, O5[r]) for r in reps])
    z = np.array([lg(oof[r]) for r in reps]); z = (z - z.mean(1, keepdims=True)) / z.std(1, keepdims=True) * base[reps].std(1, keepdims=True)
    D = {w: np.array([ap(Y, (1 - w) * base[r] + w * z[k]) for k, r in enumerate(reps)]) - b for w in WS}
    sel = [k for k, r in enumerate(reps) if r <= 2]; hold = [k for k, r in enumerate(reps) if r >= 3]
    wbest = max(WS, key=lambda w: D[w][sel].mean()) if sel else None
    res = {"name": name, "reps": reps, "own": own.mean(), "v5": v5.mean(), "base": b.mean(), "deltas": {w: D[w].mean() for w in WS},
           "wins": {w: int((D[w] > 0).sum()) for w in WS}, "w_sel": wbest, "held_out": D[wbest][hold].mean() if hold and wbest else float("nan")}
    if not quiet:
        print(f"{name}: own {own.mean():.4f} (v5 {v5.mean():.4f}, R2 base {b.mean():.4f}) | as R2 member: " +
              "  ".join(f"w{w} {D[w].mean():+.4f} ({res['wins'][w]}/{len(reps)})" for w in WS) +
              (f" | w chosen on reps0-2 = {wbest}, held-out reps3-4 {res['held_out']:+.4f}" if hold else ""), flush=True)
    return res


if __name__ == "__main__":
    for f in sys.argv[1:]: evaluate(np.load(f), f.split("oof5_")[-1][:-4])
