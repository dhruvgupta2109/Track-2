import sys, os, json, numpy as np, torch, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, "src")
import lb_infer as L
import suraj_features as SF
from gpulock import gpu_lock
S = SF.make_features(L.TE)
extra = {"amt_gt500": S.high_amount, "spend_gt2000": S.high_spend_24h, "n24_ge11": S.burst_24h, "n1_ge3": S.burst_1h,
         "risky_cnp": S.risky_cat_cnp, "risky_big": S.risky_cat_high_amount, "hour4": (L.TE.transaction_hour == 4), "age_le120": S.young_account}
L.F = np.c_[L.F, np.column_stack([np.nan_to_num(np.asarray(v, float)) for v in extra.values()])]; L.FN = L.FN + list(extra)
import lb_infer_gpu as G
G.L = L
if __name__ == "__main__":
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-6
    with gpu_lock("track2-lbinfer") as got:
        dev = "cuda" if got and torch.cuda.is_available() else "cpu"
        sim = G.make_sim(dev); A = list(range(len(L.Y_OBS))); nF = L.F.shape[1]
        def loss(th, idx):
            s, _ = sim(th, idx); return np.mean((s[idx] - L.Y_OBS[idx]) ** 2) + lam * np.sum(np.asarray(th[2:]) ** 2)
        fit = lambda idx, x0: minimize(loss, x0, args=(idx,), method="Powell", options={"maxiter": 30000, "xtol": 1e-3, "ftol": 1e-12}).x
        x0 = np.r_[-2.19, 1.0, np.zeros(nF)]; th = fit(A, x0); s1, p1 = sim(th, A)
        print(f"lam {lam}: RMSE {np.sqrt(np.mean((s1 - L.Y_OBS) ** 2)):.4f} corr {np.corrcoef(s1, L.Y_OBS)[0,1]:.3f} prev {p1.mean():.4f} a {th[0]:.2f} b {th[1]:.2f}")
        print(json.dumps({k: round(float(v), 2) for k, v in zip(L.FN, th[2:])}))
        for n, o, s in zip(L.NAMES, L.Y_OBS, s1): print(f"   {n:45s} public {o:.5f} fit {s:.5f}")
        json.dump({"theta": th.tolist(), "features": L.FN, "lam": lam}, open(f"experiments/lb_infer2_theta_{lam}.json", "w"))
        errs = []
        for i in A:
            t = fit([j for j in A if j != i], th); errs.append(sim(t, [i])[0][i] - L.Y_OBS[i])
        e = np.array(errs); print(f"LOO RMSE {np.sqrt(np.mean(e**2)):.4f}  LOO centred sd {np.std(e):.4f}  LOO corr {np.corrcoef(L.Y_OBS + e, L.Y_OBS)[0,1]:.3f}")
