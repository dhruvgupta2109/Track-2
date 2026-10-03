"""Differentiable public-score emulator: label model logit = a + sum_j w_j * z_j (standardised component logits).
Expected AP of a fixed ranking under soft labels q: sum_i q_i * cumsum(q)_i / rank_i / sum(q)."""
import sys, os, json, numpy as np, torch, pandas as pd
sys.path.insert(0, "src")
import lb_infer as L
from lb_pool import BASE
from gpulock import gpu_lock
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
COMP = sys.argv[1].split(",") if len(sys.argv) > 1 else ["v5", "sur", "k01", "mlp", "cb", "v4"]
Zc = np.vstack([lg(BASE[c]) for c in COMP]); Zc = (Zc - Zc.mean(1, keepdims=True)) / Zc.std(1, keepdims=True)

def soft_ap(q, order):  # q (n,), order list of index tensors
    out = []
    rank = torch.arange(1, len(q) + 1, device=q.device, dtype=q.dtype)
    for o in order:
        qs = q[o]; out.append((qs * qs.cumsum(0) / rank).sum() / qs.sum())
    return torch.stack(out)

if __name__ == "__main__":
    with gpu_lock("track2-lbsoft") as got:
        dev = "cuda" if got and torch.cuda.is_available() else "cpu"; print("device", dev, "components", COMP, flush=True)
        Z = torch.tensor(Zc.T, device=dev, dtype=torch.float32); Yo = torch.tensor(L.Y_OBS, device=dev, dtype=torch.float32)
        order = [torch.tensor(np.argsort(-p, kind="stable"), device=dev) for p in L.PRED]
        # sanity: soft AP vs sampled AP at the v5-only regime fitted earlier (a=-2.19 on v5 logit)
        def fit(idx, lam=1e-4, restarts=6, steps=1500):
            best = None
            for r in range(restarts):
                g = torch.Generator().manual_seed(r)
                w = (torch.randn(len(COMP), generator=g) * 0.5).to(dev).requires_grad_(); a = torch.tensor(-5.0, device=dev, requires_grad=True)
                opt = torch.optim.Adam([w, a], lr=0.03)
                for s in range(steps):
                    q = torch.sigmoid(a + Z @ w); pred = soft_ap(q, [order[i] for i in idx])
                    loss = ((pred - Yo[idx]) ** 2).mean() + lam * (w ** 2).sum()
                    opt.zero_grad(); loss.backward(); opt.step()
                if best is None or loss.item() < best[0]: best = (loss.item(), a.detach().clone(), w.detach().clone())
            return best
        A = list(range(len(L.Y_OBS)))
        for lam in [1e-5, 1e-4]:
            l, a, w = fit(A, lam); q = torch.sigmoid(a + Z @ w)
            fitted = soft_ap(q, order).cpu().numpy()
            print(f"lam {lam}: RMSE {np.sqrt(np.mean((fitted - L.Y_OBS)**2)):.4f} corr {np.corrcoef(fitted, L.Y_OBS)[0,1]:.3f} prev {q.mean().item():.4f}  a {a.item():.2f}")
            print("   weights:", {c: round(float(x), 3) for c, x in zip(COMP, w.cpu().numpy())}, flush=True)
            # LOO
            e = []
            for i in A:
                _, a_, w_ = fit([j for j in A if j != i], lam, restarts=3, steps=800)
                e.append(soft_ap(torch.sigmoid(a_ + Z @ w_), [order[i]]).item() - L.Y_OBS[i])
            e = np.array(e); print(f"   LOO RMSE {np.sqrt(np.mean(e**2)):.4f}  LOO corr {np.corrcoef(L.Y_OBS + e, L.Y_OBS)[0,1]:.3f}", flush=True)
            for n, o, f_, ee in zip(L.NAMES, L.Y_OBS, fitted, e): print(f"      {n:42s} public {o:.5f} fit {f_:.5f} loo {o+ee:.5f}")
            pmix = q.cpu().numpy(); tag = "_".join(COMP) + f"_lam{lam}"
            pd.DataFrame({"id": L.TE.id, "prediction": pmix}).to_csv(f"submissions/final_try/LBmix_{tag}.csv", index=False)
            json.dump({"a": a.item(), "w": w.cpu().tolist(), "comp": COMP, "lam": lam}, open(f"experiments/lbsoft_{tag}.json", "w"))
