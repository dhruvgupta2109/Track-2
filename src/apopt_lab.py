"""Direct average-precision optimisation of v5's linear scorer (Smooth-AP surrogate, Brown et al. 2020), warm-started from v5's
logistic-regression solution and anchored to it with an L2 penalty. Same folds as everything else (evaluate.SPLITS)."""
import sys, os, numpy as np, pandas as pd, torch, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, "src")
import models
from evaluate import TR, Y, SPLITS
from r9lab import base, embed, class_dists, EMB


def design(a, b):
    """v5's exact design matrix (fitted preprocessing from fold-train)."""
    A, B, ya, num = base(a, b)
    Ea, Eb = embed(A, B, EMB); fa, fb = class_dists(Ea, Eb, ya, 1, 1); la, lb = class_dists(Ea, Eb, ya, 0, 1)
    A["dr"] = np.log1p(fa[:, 0]) - np.log1p(la[:, 0]); B["dr"] = np.log1p(fb[:, 0]) - np.log1p(lb[:, 0]); num = num + ["dr"]
    pipe = models.lr_pipeline(0.1, num, models.LR_CAT).fit(A[models.LR_CAT + num], ya)
    ct = pipe.steps[0][1]; XA = ct.transform(A[models.LR_CAT + num]); XB = ct.transform(B[models.LR_CAT + num])
    XA = XA.toarray() if hasattr(XA, "toarray") else XA; XB = XB.toarray() if hasattr(XB, "toarray") else XB
    lr = pipe.steps[-1][1]
    return XA.astype(np.float32), XB.astype(np.float32), ya, lr.coef_[0].astype(np.float32), np.float32(lr.intercept_[0])


def smooth_ap(s, y, tau):
    pos = y > 0.5; sp = s[pos]
    d_all = torch.sigmoid((s[None, :] - sp[:, None]) / tau)            # P x N : j ranked above positive i
    d_pos = d_all[:, pos]
    r_all = 1 + d_all.sum(1) - 0.5; r_pos = 1 + d_pos.sum(1) - 0.5     # remove self term sigmoid(0)=0.5
    return (r_pos / r_all).mean()


def finetune(XA, ya, w0, b0, tau=0.05, lam=1.0, steps=150, lr=0.01, dev="cpu"):
    X = torch.tensor(XA, device=dev); y = torch.tensor(ya, dtype=torch.float32, device=dev)
    w0t = torch.tensor(w0, device=dev); w = w0t.clone().requires_grad_(); b = torch.tensor(b0, device=dev)
    s0 = X @ w0t; scale = s0.std().detach()
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        s = (X @ w) / scale
        loss = -smooth_ap(s, y, tau) + lam * ((w - w0t) ** 2).sum() / len(w0)
        opt.zero_grad(); loss.backward(); opt.step()
    return w.detach().cpu().numpy()


if __name__ == "__main__":
    cfgs = [(0.05, 1.0, 150), (0.05, 0.1, 150), (0.02, 1.0, 150), (0.1, 1.0, 300)]
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    out = {c: np.full((5, len(Y)), np.nan) for c in cfgs}; base_oof = np.full((5, len(Y)), np.nan)
    for i, (a, b) in enumerate(SPLITS[:reps * 5]):
        r = i // 5
        XA, XB, ya, w0, b0 = design(TR.iloc[a], TR.iloc[b])
        base_oof[r, b] = XB @ w0 + b0
        for c in cfgs:
            w = finetune(XA, ya, w0, b0, *c); out[c][r, b] = XB @ w
        print(f"fold {i}: v5-LR {ap(Y[b], base_oof[r, b]):.4f} | " + "  ".join(f"tau{c[0]}/lam{c[1]}/{c[2]} {ap(Y[b], out[c][r, b]):.4f}" for c in cfgs), flush=True)
    for c in cfgs:
        sc = 1 / (1 + np.exp(-out[c])); np.save(f"experiments/oof5_apopt_t{c[0]}_l{c[1]}_s{c[2]}.npy", sc)
        print(f"cfg {c}: " + "  ".join(f"rep{r} {ap(Y, out[c][r]):.4f} vs LR {ap(Y, base_oof[r]):.4f}" for r in range(reps)))
