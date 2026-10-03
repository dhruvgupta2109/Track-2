"""Neural branch: (1) regularised MLP on the v5/E1 design matrix; (2) supervised-contrastive embedding -> kNN ratio fed to LR.
Fast screen folds (2 reps), paired vs v5. Uses the GPU under the shared lock (falls back to CPU if the lock is held)."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import average_precision_score as ap
from scipy.stats import rankdata
from r9lab import *
from gpulock import gpu_lock
import models

E1 = [c for c in EMB if c != "amt_rel_avg"] + ["burst"]


def design(a, b):
    A, B, ya, num = base(a, b)
    Ea, Eb = embed(A, B, E1); fa, fb = class_dists(Ea, Eb, ya, 1, 1); la, lb = class_dists(Ea, Eb, ya, 0, 1)
    A["dr"] = np.log1p(fa[:, 0]) - np.log1p(la[:, 0]); B["dr"] = np.log1p(fb[:, 0]) - np.log1p(lb[:, 0]); num = num + ["dr"]
    ct = models.lr_pipeline(0.1, num, models.LR_CAT).steps[0][1].fit(A[models.LR_CAT + num])
    Xa, Xb = ct.transform(A[models.LR_CAT + num]), ct.transform(B[models.LR_CAT + num])
    Xa = Xa.toarray() if hasattr(Xa, "toarray") else Xa; Xb = Xb.toarray() if hasattr(Xb, "toarray") else Xb
    return np.asarray(Xa, np.float32), np.asarray(Xb, np.float32), ya, A, B, Ea, Eb


class MLP(nn.Module):
    def __init__(s, d, h=64, p=0.3):
        super().__init__(); s.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(p), nn.Linear(h // 2, 1))
    def forward(s, x): return s.f(x).squeeze(-1)


def train_mlp(Xa, ya, Xb, dev, seed, epochs=60, wd=1e-3, lr=1e-3):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ya)); nv = len(ya) // 8; va, trn = idx[:nv], idx[nv:]
    m = MLP(Xa.shape[1]).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(Xa[trn], device=dev); yt = torch.tensor(ya[trn], dtype=torch.float32, device=dev)
    Xv = torch.tensor(Xa[va], device=dev); best, state, bad = -1, None, 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(trn), device=dev)
        for i in range(0, len(trn), 512):
            j = perm[i:i + 512]; opt.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(m(Xt[j]), yt[j]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): s = ap(ya[va], m(Xv).cpu().numpy())
        if s > best: best, state, bad = s, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10: break
    m.load_state_dict(state); m.eval()
    with torch.no_grad(): return torch.sigmoid(m(torch.tensor(Xb, device=dev))).cpu().numpy()


class Enc(nn.Module):
    def __init__(s, d, k=8):
        super().__init__(); s.f = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, k))
    def forward(s, x): return nn.functional.normalize(s.f(x), dim=-1)


def supcon_embed(Ea, ya, Eb, dev, seed, hard_idx, epochs=150, tau=0.2):
    """batches = all frauds + equal number of hard legit + random legit; supervised contrastive loss."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Enc(Ea.shape[1]).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-3)
    F = np.where(ya == 1)[0]; L = np.where(ya == 0)[0]
    X = torch.tensor(Ea, dtype=torch.float32, device=dev)
    for ep in range(epochs):
        b = np.r_[F, rng.choice(hard_idx, min(len(F), len(hard_idx)), replace=False), rng.choice(L, 2 * len(F), replace=False)]
        z = m(X[b]); lab = torch.tensor(ya[b], device=dev)
        sim = z @ z.T / tau; sim = sim - torch.eye(len(b), device=dev) * 1e9
        pos = (lab[:, None] == lab[None, :]).float() - torch.eye(len(b), device=dev)
        logp = sim - torch.logsumexp(sim, 1, keepdim=True)
        loss = -((logp * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(X).cpu().numpy(), m(torch.tensor(Eb, dtype=torch.float32, device=dev)).cpu().numpy()


def main():
    with gpu_lock("track2-nn") as got:
        dev = "cuda" if (got and torch.cuda.is_available()) else "cpu"; print("device", dev, flush=True)
        reps = 2; folds = SPLITS[: 5 * reps]
        oof_mlp = np.zeros((reps, len(Y))); oof_sc = np.zeros((reps, len(Y))); oof_scdr = np.zeros((reps, len(Y)))
        for i, (ia, ib) in enumerate(folds):
            a, b = TR.iloc[ia], TR.iloc[ib]
            Xa, Xb, ya, A, B, Ea, Eb = design(a, b)
            oof_mlp[i // 5, ib] = np.mean([train_mlp(Xa, ya, Xb, dev, s) for s in range(5)], 0)
            # hard legit: legit rows whose own E-space nearest-fraud ratio is in the lowest 10%
            hard = np.where((ya == 0) & (A.dr.values <= np.quantile(A.dr.values[ya == 0], 0.10)))[0]
            Za, Zb = supcon_embed(Ea, ya, Eb, dev, 0, hard)
            fa, fb = class_dists(Za, Zb, ya, 1, 1); la, lb = class_dists(Za, Zb, ya, 0, 1)
            A["dr_sc"] = np.log1p(fa[:, 0]) - np.log1p(la[:, 0]); B["dr_sc"] = np.log1p(fb[:, 0]) - np.log1p(lb[:, 0])
            _, _, _, num = base(a.iloc[:50], b.iloc[:5])
            oof_scdr[i // 5, ib] = fit_lr(A, B, ya, num + ["dr", "dr_sc"])
            print("fold", i, "done", flush=True)
    np.save("experiments/oofscreen_mlp.npy", oof_mlp); np.save("experiments/oofscreen_scdr.npy", oof_scdr)
    s5 = np.mean([ap(Y, O5[r]) for r in range(reps)]); h5 = np.mean([ap(Y, O5[r], sample_weight=W_HN) for r in range(reps)])
    rep = lambda nm, o: print(f"{nm:40s} std {np.mean([ap(Y, o[r]) for r in range(reps)]):.4f} (d {np.mean([ap(Y, o[r]) for r in range(reps)]) - s5:+.4f})  "
                              f"hnW {np.mean([ap(Y, o[r], sample_weight=W_HN) for r in range(reps)]):.4f} (d {np.mean([ap(Y, o[r], sample_weight=W_HN) for r in range(reps)]) - h5:+.4f})", flush=True)
    rep("MLP 5-seed (standalone)", oof_mlp); rep("LR + E1 dr + supcon dr", oof_scdr)
    rk = lambda x: np.vstack([rankdata(r) / len(r) for r in x])
    for w in [0.05, 0.1, 0.2, 0.3]:
        rep(f"blend rank {1-w:.2f} v5 + {w} MLP", (1 - w) * rk(O5[:reps]) + w * rk(oof_mlp))
        rep(f"blend prob {1-w:.2f} v5 + {w} MLP", (1 - w) * O5[:reps] + w * oof_mlp)


if __name__ == "__main__":
    main()
