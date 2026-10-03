"""Local emulator for the burst-by-country cells. All six scored files share R2op's ranking outside bursts and differ only by
logit shifts on three burst cells (MY/ID/SG bursts, young other-country bursts, old other-country bursts).
Label model: q_i = sigmoid(logit(s * p_i) + delta_c) on burst cells (p = R2op ranking probability, pre operating shift), q_i = s * p_i elsewhere.
Fit (delta_MYIDSG, delta_oy, delta_oo) to the six public scores (differences vs R2op), check leave-one-out, then optimise the next shifts."""
import sys, itertools, numpy as np, pandas as pd, joblib, warnings; warnings.filterwarnings("ignore")
from scipy.optimize import minimize
sys.path.insert(0, "src")
from r2_runtime import predict_r2

te = pd.read_csv("test.csv")
_, det = predict_r2(te, joblib.load("../Track2_R2op_package/model.pkl"), return_details=True)
P = det["ranking"]; lg = lambda x: np.log(np.clip(x, 1e-12, 1 - 1e-12) / (1 - np.clip(x, 1e-12, 1 - 1e-12))); sg = lambda z: 1 / (1 + np.exp(-z))
burst = (te.transactions_last_24h >= 16).values; young = (te.account_age <= 365).values
pu = te.country.isin(["MY", "ID", "SG"]).values & burst; oth = te.country.notna().values & ~te.country.isin(["MY", "ID", "SG"]).values & burst
CELLS = {"MYIDSG": pu, "oth_young": oth & young, "oth_old": oth & ~young}
FILES = {"R2op": ((0, 0, 0), 0.23434), "B1": ((-0.25, 0.25, 0.25), 0.24016), "B2a": ((-0.25, 0.5, 0.5), 0.24094),
         "B2b": ((-0.5, 0.25, 0.25), 0.23904), "C1": ((-0.25, 0.75, 0.25), 0.23702), "C2": ((0, 0.5, 0.5), 0.24174)}
Z0 = lg(P)


def ranking(sh):
    z = Z0.copy()
    for (k, m), d in zip(CELLS.items(), sh): z[m] += d
    return np.argsort(-z, kind="stable")


ORD = {k: ranking(v[0]) for k, v in FILES.items()}; OBS = np.array([v[1] for v in FILES.values()])


def soft_ap(q, order):
    qs = q[order]; return float((qs * np.cumsum(qs) / np.arange(1, len(qs) + 1)).sum() / qs.sum())


def qmodel(delta, s):
    z = lg(np.clip(s * P, 1e-9, 0.999))
    for (k, m), d in zip(CELLS.items(), delta): z[m] += d
    return sg(z)


def pred(delta, s, orders):
    q = qmodel(delta, s); return np.array([soft_ap(q, o) for o in orders])


def fit(idx, s, x0=(0, 0, 0)):
    names = list(FILES); orders = [ORD[n] for n in names]
    def loss(d):
        e = pred(d, s, orders); r = (e - e[0]) - (OBS - OBS[0]); return float(np.sum(r[idx] ** 2)) + 1e-6 * float(np.sum(np.square(d)))
    best = min((minimize(loss, np.array(x0) + j, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-10, "maxiter": 600}) for j in (0, 0.5, -0.5)), key=lambda r: r.fun)
    return best.x


if __name__ == "__main__":
    names = list(FILES); orders = [ORD[n] for n in names]
    for s in (1.0, 1.3, 1.6):
        d = fit(list(range(1, 6)), s); e = pred(d, s, orders)
        print(f"s={s}: fitted cell deltas MYIDSG {d[0]:+.2f} oth_young {d[1]:+.2f} oth_old {d[2]:+.2f} | level R2op pred {e[0]:.4f} (obs 0.23434)")
        print("   deltas vs R2op  pred | obs : " + "  ".join(f"{n} {e[i] - e[0]:+.4f}|{OBS[i] - OBS[0]:+.4f}" for i, n in enumerate(names) if i))
        loo = []
        for k in range(1, 6):
            dk = fit([j for j in range(1, 6) if j != k], s, d); ek = pred(dk, s, orders); loo.append((names[k], ek[k] - ek[0], OBS[k] - OBS[0]))
        print("   LOO (pred|obs): " + "  ".join(f"{n} {p_:+.4f}|{o_:+.4f}" for n, p_, o_ in loo), f" LOO RMSE {np.sqrt(np.mean([(p_ - o_) ** 2 for _, p_, o_ in loo])):.4f}")
        # best next shifts under the fitted model (grid near the explored region)
        grid = [(a, y, o) for a in (-0.25, 0, 0.25, 0.5) for y in (0.25, 0.5, 0.75) for o in (0.25, 0.5, 0.75, 1.0)]
        q = qmodel(d, s); vals = sorted(((soft_ap(q, ranking(g)), g) for g in grid), reverse=True)
        c2 = soft_ap(q, ranking((0, 0.5, 0.5)))
        print("   top next shifts (pred gain vs C2): " + "  ".join(f"{g} {v - c2:+.4f}" for v, g in vals[:6]), flush=True)
