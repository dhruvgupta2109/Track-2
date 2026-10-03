"""Deep tabular models (research): ModernNCA-style soft-NN, TabM-style packed MLP ensemble, large MLP with PLR embeddings.
Input = v5 design matrix (one-hot categoricals + standardised numerics incl. v5 kNN ratio)."""
import sys, os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, os.path.dirname(__file__))
import models
from r9lab import base, embed, class_dists, EMB


def design_v5(a, b, with_dr=True):
    A, B, ya, num = base(a, b)
    if with_dr:
        Ea, Eb = embed(A, B, EMB); fa, fb = class_dists(Ea, Eb, ya, 1, 1); la, lb = class_dists(Ea, Eb, ya, 0, 1)
        A["dr"] = np.log1p(fa[:, 0]) - np.log1p(la[:, 0]); B["dr"] = np.log1p(fb[:, 0]) - np.log1p(lb[:, 0]); num = num + ["dr"]
    ct = models.lr_pipeline(0.1, num, models.LR_CAT).steps[0][1].fit(A[models.LR_CAT + num])
    Xa, Xb = ct.transform(A[models.LR_CAT + num]), ct.transform(B[models.LR_CAT + num])
    Xa = Xa.toarray() if hasattr(Xa, "toarray") else Xa; Xb = Xb.toarray() if hasattr(Xb, "toarray") else Xb
    n_num = len(num)
    return np.asarray(Xa, np.float32), np.asarray(Xb, np.float32), ya.astype(np.float32), n_num


class PLR(nn.Module):
    """periodic-linear-relu numeric embedding (Gorishniy et al. 2022) for the last n_num columns."""
    def __init__(s, n_num, k=8, d=8, sigma=0.5):
        super().__init__(); s.c = nn.Parameter(torch.randn(n_num, k) * sigma); s.lin = nn.Linear(2 * k, d); s.n = n_num; s.d = d
    def forward(s, x):
        xn = x[:, -s.n:]; v = 2 * np.pi * s.c[None] * xn[..., None]
        e = Fn.relu(s.lin(torch.cat([torch.cos(v), torch.sin(v)], -1)))
        return torch.cat([x[:, :-s.n], e.flatten(1)], 1)
    def out_dim(s, d_in): return d_in - s.n + s.n * s.d


class MLPNet(nn.Module):
    def __init__(s, d_in, n_num, h=256, depth=3, p=0.2, plr=True):
        super().__init__(); s.plr = PLR(n_num) if plr else None; d = s.plr.out_dim(d_in) if plr else d_in
        L = []
        for _ in range(depth): L += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]; d = h
        s.body = nn.Sequential(*L); s.head = nn.Linear(h, 1)
    def forward(s, x):
        if s.plr is not None: x = s.plr(x)
        return s.head(s.body(x)).squeeze(-1)


class TabMmini(nn.Module):
    """k independent MLPs evaluated in parallel (packed ensemble); prediction = mean prob."""
    def __init__(s, d_in, n_num, k=16, h=128, depth=2, p=0.1, plr=True):
        super().__init__(); s.k = k; s.plr = PLR(n_num) if plr else None; d = s.plr.out_dim(d_in) if plr else d_in
        s.W = nn.ParameterList(); s.bias = nn.ParameterList(); dims = [d] + [h] * depth
        for i in range(depth):
            s.W.append(nn.Parameter(torch.randn(k, dims[i], dims[i + 1]) / np.sqrt(dims[i]))); s.bias.append(nn.Parameter(torch.zeros(k, 1, dims[i + 1])))
        s.Wo = nn.Parameter(torch.randn(k, h, 1) / np.sqrt(h)); s.bo = nn.Parameter(torch.zeros(k, 1, 1)); s.p = p
    def forward(s, x):
        if s.plr is not None: x = s.plr(x)
        z = x[None].expand(s.k, -1, -1)
        for W, b in zip(s.W, s.bias): z = Fn.dropout(Fn.relu(torch.bmm(z, W) + b), s.p, s.training)
        return (torch.bmm(z, s.Wo) + s.bo).squeeze(-1)          # (k, n) logits


class ModernNCA(nn.Module):
    def __init__(s, d_in, n_num, dim=64, h=128, p=0.1, plr=True, linear=False):
        super().__init__(); s.plr = PLR(n_num) if plr else None; d = s.plr.out_dim(d_in) if plr else d_in
        s.enc = nn.Linear(d, dim) if linear else nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, dim))
        s.log_t = nn.Parameter(torch.zeros(()))
    def embed(s, x): return s.enc(s.plr(x) if s.plr is not None else x)
    def logits(s, zq, zc, yc, self_mask=None):
        d = torch.cdist(zq, zc) * torch.exp(-s.log_t)
        s_ = -d
        if self_mask is not None: s_ = s_.masked_fill(self_mask, -1e9)
        f = yc > 0.5
        return torch.logsumexp(s_[:, f], 1) - torch.logsumexp(s_[:, ~f], 1)


def _val_split(n, seed, frac=0.125):
    idx = np.random.default_rng(seed).permutation(n); nv = int(n * frac); return idx[nv:], idx[:nv]


def train_net(kind, Xa, ya, Xb, n_num, dev, seed=0, epochs=80, lr=1e-3, wd=1e-4, bs=512, patience=12, **kw):
    torch.manual_seed(seed); trn, va = _val_split(len(ya), seed)
    X = torch.tensor(Xa, device=dev); Y = torch.tensor(ya, device=dev); Xq = torch.tensor(Xb, device=dev)
    d_in = Xa.shape[1]
    m = {"mlp": MLPNet, "tabm": TabMmini, "nca": ModernNCA}[kind](d_in, n_num, **kw).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    trn_t = torch.tensor(trn, device=dev)

    def predict(Xin, cand_idx):
        m.eval()
        with torch.no_grad():
            if kind == "mlp": return torch.sigmoid(m(Xin)).cpu().numpy()
            if kind == "tabm": return torch.sigmoid(m(Xin)).mean(0).cpu().numpy()
            zc = m.embed(X[cand_idx]); out = []
            for i in range(0, len(Xin), 2048): out.append(torch.sigmoid(m.logits(m.embed(Xin[i:i + 2048]), zc, Y[cand_idx])))
            return torch.cat(out).cpu().numpy()

    best, state, bad = -1, None, 0
    for ep in range(epochs):
        m.train(); perm = trn_t[torch.randperm(len(trn), device=dev)]
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            if kind == "mlp": loss = Fn.binary_cross_entropy_with_logits(m(X[j]), Y[j])
            elif kind == "tabm": lg = m(X[j]); loss = Fn.binary_cross_entropy_with_logits(lg, Y[j][None].expand_as(lg))
            else:
                cand = trn_t[torch.rand(len(trn), device=dev) < 0.3]
                zq, zc = m.embed(X[j]), m.embed(X[cand]); mask = j[:, None] == cand[None, :]
                loss = Fn.binary_cross_entropy_with_logits(m.logits(zq, zc, Y[cand], mask), Y[j])
            opt.zero_grad(); loss.backward(); opt.step()
        s = ap(ya[va], predict(X[torch.tensor(va, device=dev)], trn_t))
        if s > best: best, state, bad = s, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(state)
    return predict(Xq, torch.arange(len(ya), device=dev))
