import sys, os, json, numpy as np, torch
from scipy.optimize import minimize
sys.path.insert(0, "src")
import lb_infer as L
from gpulock import gpu_lock
R = 256
def setup(dev):
    g = torch.Generator(device="cpu").manual_seed(0)
    U = torch.rand(R, len(L.Z0), generator=g).to(dev); M = (torch.rand(R, len(L.Z0), generator=g) < 0.5).float().to(dev)
    order = [torch.tensor(np.argsort(-p, kind="stable"), device=dev) for p in L.PRED]
    return U, M, order, torch.tensor(L.Z0, device=dev, dtype=torch.float32), torch.tensor(L.F, device=dev, dtype=torch.float32)
def make_sim(dev):
    U, M, order, Z0, F = setup(dev)
    def sim(theta, idx):
        th = torch.tensor(theta, device=dev, dtype=torch.float32)
        p = torch.sigmoid(th[0] + th[1] * Z0 + F @ th[2:]); Y = (U < p).float() * M
        out = np.zeros(len(L.Y_OBS))
        for i in idx:
            ys, ms = Y[:, order[i]], M[:, order[i]]
            prec = ys.cumsum(1) / ms.cumsum(1).clamp(min=1)
            out[i] = ((ys * prec).sum(1) / ys.sum(1).clamp(min=1)).mean().item()
        return out, p.cpu().numpy()
    return sim
if __name__ == "__main__":
    with gpu_lock("track2-lbinfer") as got:
        dev = "cuda" if got and torch.cuda.is_available() else "cpu"; print("device", dev)
        sim = make_sim(dev); A = list(range(len(L.Y_OBS))); nF = L.F.shape[1]
        def loss(th, idx, lam=2e-4):
            s, _ = sim(th, idx); return np.mean((s[idx] - L.Y_OBS[idx]) ** 2) + lam * np.sum(np.asarray(th[2:]) ** 2)
        def fit(idx, x0):
            return minimize(loss, x0, args=(idx,), method="Powell", options={"maxiter": 20000, "xtol": 1e-3, "ftol": 1e-10}).x
        base = np.r_[0.0, 1.0, np.zeros(nF)]; s0, p0 = sim(base, A)
        print("baseline: RMSE %.4f corr %.3f prev %.4f" % (np.sqrt(np.mean((s0 - L.Y_OBS) ** 2)), np.corrcoef(s0, L.Y_OBS)[0, 1], p0.mean()))
        # intercept/slope only
        def fit_ab(idx, x0):
            f = lambda ab: loss(np.r_[ab, np.zeros(nF)], idx); return np.r_[minimize(f, x0[:2], method="Powell").x, np.zeros(nF)]
        tab = fit_ab(A, base); sab, pab = sim(tab, A)
        print("a,b only: RMSE %.4f corr %.3f a %.3f b %.3f prev %.4f" % (np.sqrt(np.mean((sab - L.Y_OBS) ** 2)), np.corrcoef(sab, L.Y_OBS)[0, 1], tab[0], tab[1], pab.mean()))
        th = fit(A, tab); s1, p1 = sim(th, A)
        print("full: RMSE %.4f corr %.3f prev %.4f a %.3f b %.3f" % (np.sqrt(np.mean((s1 - L.Y_OBS) ** 2)), np.corrcoef(s1, L.Y_OBS)[0, 1], p1.mean(), th[0], th[1]))
        print(json.dumps({k: round(float(v), 3) for k, v in zip(L.FN, th[2:])}))
        for n, o, a_, s in zip(L.NAMES, L.Y_OBS, sab, s1): print(f"   {n:45s} public {o:.5f}  ab-fit {a_:.5f}  full-fit {s:.5f}")
        json.dump({"theta": th.tolist(), "theta_ab": tab.tolist(), "features": L.FN}, open("experiments/lb_infer_theta.json", "w"))
        e_ab, e_full = [], []
        for i in A:
            rest = [j for j in A if j != i]
            t1 = fit(rest, th); t0 = fit_ab(rest, tab)
            e_full.append(sim(t1, [i])[0][i] - L.Y_OBS[i]); e_ab.append(sim(t0, [i])[0][i] - L.Y_OBS[i])
            print(f"LOO {L.NAMES[i]:45s} obs {L.Y_OBS[i]:.5f}  ab {L.Y_OBS[i]+e_ab[-1]:.5f}  full {L.Y_OBS[i]+e_full[-1]:.5f}", flush=True)
        print("LOO RMSE: a,b-only %.4f   full regime %.4f" % (np.sqrt(np.mean(np.square(e_ab))), np.sqrt(np.mean(np.square(e_full)))))
