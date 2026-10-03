"""Label model = a + sum_j b_j * logit(component_j). Fitted to the 19 public scores; LOO check; emits the implied submission."""
import sys, os, json, numpy as np, torch, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, "src")
import lb_infer as L
import lb_infer_gpu as G
from lb_pool import BASE
from gpulock import gpu_lock
from scipy.stats import rankdata
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
COMP = ["v5", "sur", "k01", "mlp", "cb"]
Zc = np.vstack([lg(BASE[c]) for c in COMP]); Zc = (Zc - Zc.mean(1, keepdims=True)) / Zc.std(1, keepdims=True)
L.Z0 = np.zeros(len(L.Z0)); L.F = Zc.T                     # reuse the simulator: theta = [a, unused b, weights...]
L.FN = COMP
if __name__ == "__main__":
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-6
    with gpu_lock("track2-lbinfer3") as got:
        dev = "cuda" if got and torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
        sim = G.make_sim(dev); A = list(range(len(L.Y_OBS)))
        def loss(th, idx):
            t = np.r_[th[0], 0.0, th[1:]]; s, _ = sim(t, idx); return np.mean((s[idx] - L.Y_OBS[idx]) ** 2) + lam * np.sum(th[1:] ** 2)
        fit = lambda idx, x0: minimize(loss, x0, args=(idx,), method="Powell", options={"maxiter": 20000, "xtol": 1e-3, "ftol": 1e-12}).x
        best = None
        for x0 in [np.r_[-6.0, 1.5, 0, 0, 0, 0], np.r_[-6.0, 0.8, 0.8, 0, 0, 0], np.r_[-6.0, 0.5, 0.5, 0.5, 0.3, 0]]:
            th = fit(A, x0); l = loss(th, A)
            if best is None or l < best[0]: best = (l, th)
        th = best[1]; s1, p1 = sim(np.r_[th[0], 0.0, th[1:]], A)
        print(f"lam {lam}: RMSE {np.sqrt(np.mean((s1 - L.Y_OBS) ** 2)):.4f} corr {np.corrcoef(s1, L.Y_OBS)[0,1]:.3f} prev {p1.mean():.4f}")
        print("weights (on standardised logits):", json.dumps({c: round(float(w), 3) for c, w in zip(COMP, th[1:])}), flush=True)
        for n, o, s in zip(L.NAMES, L.Y_OBS, s1): print(f"   {n:45s} public {o:.5f} fit {s:.5f}")
        z = Zc.T @ th[1:]; p = 1 / (1 + np.exp(-z))
        pd.DataFrame({"id": L.TE.id, "prediction": p}).to_csv("submissions/final_try/LB1_fitted_mix.csv", index=False)
        json.dump({"theta": th.tolist(), "comp": COMP}, open("experiments/lb_infer3_theta.json", "w"))
        e = []
        for i in A:
            t = fit([j for j in A if j != i], th); e.append(sim(np.r_[t[0], 0.0, t[1:]], [i])[0][i] - L.Y_OBS[i])
            print(f"LOO {L.NAMES[i]:45s} obs {L.Y_OBS[i]:.5f} pred {L.Y_OBS[i]+e[-1]:.5f}", flush=True)
        e = np.array(e); print(f"LOO RMSE {np.sqrt(np.mean(e**2)):.4f} LOO corr {np.corrcoef(L.Y_OBS + e, L.Y_OBS)[0,1]:.3f}")
