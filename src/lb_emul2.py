"""History emulator v2: one label model fitted to every public score (48 files).
Non-template rows: logit q = a + b1 z_v5 + b2 z_poly + b3 z_galr + sum_g d_g 1[g]; planted boxes T1/T2/T3: q = sigmoid(c_T) (labels random inside a box).
Expected AP of each fixed ranking under soft labels q (differentiable); fit by least squares + ridge on d. Validated by a time-ordered holdout."""
import sys, json, numpy as np, pandas as pd, torch, joblib
sys.path.insert(0, "src")
from lb_history import load
from templates import template_masks
from r2_runtime import predict_r2, make_features
from constants import fit_constants
from gpulock import gpu_lock

te = pd.read_csv("test.csv"); tr = pd.read_csv("train.csv")
_, det = predict_r2(te, joblib.load("../Track2_R2op_package/model.pkl"), return_details=True)
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
COMP = np.vstack([lg(det["v5"]), lg(det["poly"]), lg(det["galr"])])
D = make_features(te.copy(), fit_constants(tr)); M = template_masks(te)
a_ = te.account_age.values; b = (te.transactions_last_24h >= 16).values; nd = (te.new_device == 1).values; h = te.transaction_hour.values
young = a_ <= 365; ndold = nd & (a_ > 365) & ~b; amt = te.transaction_amount.values
hvc = te.merchant_category.isin(["luxury", "cash_transfer", "electronics", "travel"]).values
lowv = ~((te.transactions_last_1h >= 3) | (te.transactions_last_24h >= 12)).values
hvout = hvc & (amt >= 1000) & (a_ > 365) & ~b & ~ndold & ~M["T2"]
GROUPS = {"burst_young": b & young, "burst_old": b & ~young, "nd_old_base": ndold & ~M["T1"], "hv_out": hvout,
          "nd_young_night_lowvel": nd & (a_ < 180) & np.isin(h, [22, 23, 0, 1, 2, 3, 4]) & lowv & ~b,
          "fast_young_nd": nd & (a_ <= 90) & (te.transactions_last_1h >= 3).values & ~b,
          "travel_rest": (te.merchant_category == "travel").values & ~b & ~ndold & ~M["T2"] & ~hvout,
          "bigold_rest": (D.amt_rel_cat > 1).values & (a_ > 1500) & ~b & ~ndold & ~M["T2"] & ~hvout,
          "clean_hours": np.isin(h, [5, 6, 7, 21, 22]), "foreign_rest": (te.country != "SG").values & te.country.notna().values & ~b & ~ndold & ~M["T2"] & ~hvout}
TMPL = ["T1", "T2", "T3"]
G = np.vstack([GROUPS[k] for k in GROUPS]).astype(float); TM = np.vstack([M[k] for k in TMPL]).astype(float)
in_tmpl = TM.sum(0) > 0
NAMES, PRED, OBS = load(check=False)


def soft_ap(q, orders):
    rank = torch.arange(1, len(q) + 1, device=q.device, dtype=q.dtype)
    return torch.stack([(q[o] * q[o].cumsum(0) / rank).sum() / q.sum() for o in orders])


def build(dev):
    T = lambda x: torch.tensor(x, device=dev, dtype=torch.float32)
    return T(COMP.T), T(G.T), T(TM.T), T(in_tmpl.astype(float)), [torch.tensor(np.argsort(-p, kind="stable"), device=dev) for p in PRED], T(OBS)


def q_of(p, Zc, Gt, TMt, IT):
    base = torch.sigmoid(p["a"] + Zc @ p["b"] + Gt @ p["d"])
    tq = (TMt * torch.sigmoid(p["c"])).sum(1)
    return base * (1 - IT) + tq * IT


def fit(idx, Zc, Gt, TMt, IT, orders, Yo, lam=3e-4, restarts=5, steps=1500, dev="cpu", seed0=0):
    best = None
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed0 + r)
        p = {"a": torch.tensor(-4.0 + 0.3 * torch.randn(1, generator=g).item(), device=dev),
             "b": (torch.tensor([0.6, 0.2, 0.2]) + 0.1 * torch.randn(3, generator=g)).to(dev),
             "d": (0.2 * torch.randn(G.shape[0], generator=g)).to(dev), "c": (torch.full((3,), -4.0) + 0.5 * torch.randn(3, generator=g)).to(dev)}
        for v in p.values(): v.requires_grad_()
        opt = torch.optim.Adam(p.values(), lr=0.03)
        for s in range(steps):
            q = q_of(p, Zc, Gt, TMt, IT); pred = soft_ap(q, [orders[i] for i in idx])
            loss = ((pred - Yo[idx]) ** 2).mean() + lam * (p["d"] ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        if best is None or loss.item() < best[0]: best = (loss.item(), {k: v.detach().clone() for k, v in p.items()})
    return best[1]


if __name__ == "__main__":
    with gpu_lock("track2-emul2") as got:
        dev = "cuda" if got and torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
        Zc, Gt, TMt, IT, orders, Yo = build(dev); n = len(OBS); allidx = list(range(n))
        late = [NAMES.index(k) for k in NAMES if k.startswith(("C1_", "C7_", "R2op", "R3op"))]
        early = [i for i in allidx if i not in late]
        for lam in (1e-4, 3e-4, 1e-3):
            p = fit(early, Zc, Gt, TMt, IT, orders, Yo, lam, dev=dev); pr = soft_ap(q_of(p, Zc, Gt, TMt, IT), orders).detach().cpu().numpy()
            q30 = NAMES.index("predictions_Q30.csv")
            print(f"lam {lam}: early-fit RMSE {np.sqrt(np.mean((pr[early] - OBS[early]) ** 2)):.4f}; TIME HOLDOUT (delta vs Q30, pred | obs): " +
                  "  ".join(f"{NAMES[i][:6]} {pr[i] - pr[q30]:+.4f}|{OBS[i] - OBS[q30]:+.4f}" for i in late), flush=True)
        p = fit(allidx, Zc, Gt, TMt, IT, orders, Yo, 3e-4, dev=dev, restarts=8)
        q = q_of(p, Zc, Gt, TMt, IT); pr = soft_ap(q, orders).detach().cpu().numpy()
        print(f"ALL-fit RMSE {np.sqrt(np.mean((pr - OBS) ** 2)):.4f} corr {np.corrcoef(pr, OBS)[0, 1]:.3f}; prevalence {q.mean().item():.4f}")
        print("b (v5, poly, galr):", np.round(p["b"].cpu().numpy(), 3), " a", round(p["a"].item(), 2))
        print("group logit effects:", {k: round(float(v), 2) for k, v in zip(GROUPS, p["d"].cpu().numpy())})
        print("template fraud rates:", {k: round(float(v), 4) for k, v in zip(TMPL, torch.sigmoid(p["c"]).cpu().numpy())})
        np.save("experiments/emul2_q.npy", q.detach().cpu().numpy())
        json.dump({k: v.cpu().tolist() for k, v in p.items()}, open("experiments/emul2_params.json", "w"))
        for nme, o, f_ in sorted(zip(NAMES, OBS, pr), key=lambda t: t[1]): print(f"   {nme:42s} public {o:.5f} fit {f_:.5f}")
