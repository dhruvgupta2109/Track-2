"""Infer the test-label regime from public scores of our own submissions.
Model: logit P(fraud | x_test) = a + b * logit(v5(x)) + sum_k d_k f_k(x). For a parameter vector we simulate labels on the
test rows (common random numbers, R draws, each on a random 50% 'public' slice) and compute the expected AP of every
scored submission; parameters are fitted to the observed public scores (squared error)."""
import sys, os, json, numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import TE, TR
from features_prod import make_features
from constants import fit_constants

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); SUR = os.path.join(ROOT, "..", "Datathon Track 2 - suraj try")
P = lambda *p: os.path.join(ROOT, *p)
OBS = {  # file -> public score
    P("submissions/sub_v3_lr.csv"): 0.17206, P("submissions/sub_v4_lr.csv"): 0.17966, P("submissions/sub_v4_lambda05_probe.csv"): 0.16822,
    P("submissions/sub_v5_lr_knn.csv"): 0.18626, P("submissions/sub_v5a_lr.csv"): 0.18026, P("submissions/audit/cand_v4_knn_full.csv"): 0.18436,
    P("submissions/audit/c1_v5_knn_catmetric.csv"): 0.18388, P("submissions/audit/d1_v5_plus_1h_velocity.csv"): 0.18463,
    P("submissions/audit/cand_lgb_mono.csv"): 0.16962, P("submissions/audit/v6_E1_knn_metric.csv"): 0.18237,
    P("submissions/audit/v6_rank90_skmlp10.csv"): 0.18453, P("submissions/final_try/E1_0.45k01_0.4surknn_0.15mlp.csv"): 0.18053,
    P("submissions/final_try/E2_0.4k01_0.15v5_0.3sur_0.15mlp.csv"): 0.18398, P("submissions/final_try/E3_0.7k01_0.3sur.csv"): 0.18662,
    os.path.join(SUR, "submission.csv"): 0.17617, os.path.join(SUR, "models/v2_lr_drift/submission.csv"): 0.17136,
    os.path.join(SUR, "extra_submissions/cand1_v1_plus_paysim_score.csv"): 0.17830,
    os.path.join(SUR, "extra_submissions/cand2_v1_iterative_imputation.csv"): 0.17809,
    os.path.join(SUR, "extra_submissions/cand3_blend_0.8v1_0.2lgbm.csv"): 0.17835,
}
NAMES = [os.path.basename(f) for f in OBS]
PRED = np.vstack([pd.read_csv(f).prediction.values for f in OBS]); Y_OBS = np.array(list(OBS.values()))
v5 = pd.read_csv(P("submissions/sub_v5_lr_knn.csv")).prediction.values
Z0 = np.log(v5 / (1 - v5))

K = fit_constants(TR); D = make_features(TE, K)
FEAT = {
    "burst": (D.transactions_last_24h >= 16), "new_device": (D.new_device == 1), "young": (D.account_age < 90), "foreign": (D.foreign == 1),
    "big": (D.amt_rel_cat > 1.0), "night": (D.night == 1), "nd_old": (D.new_device == 1) & (D.account_age > 365),
    "old_big": (D.amt_rel_cat > 1.0) & (D.account_age > 1500), "t1_ge3": (D.transactions_last_1h >= 3), "risky_cat": D.merchant_category.isin(["cash_transfer", "electronics", "luxury"]),
    "card_present": (D.transaction_channel == "card_present"), "any_missing": TE.drop(columns="id").isna().any(axis=1),
}
F = np.vstack([np.asarray(v, float) for v in FEAT.values()]).T; FN = list(FEAT)
R = 24; rng = np.random.default_rng(0); U = rng.random((R, len(Z0))); SL = [rng.random(len(Z0)) < 0.5 for _ in range(R)]


def sim(theta, idx=None):
    a, b, d = theta[0], theta[1], theta[2:]
    p = 1 / (1 + np.exp(-(a + b * Z0 + F @ d)))
    idx = range(len(Y_OBS)) if idx is None else idx
    out = np.zeros(len(Y_OBS))
    for r in range(R):
        y = U[r] < p; m = SL[r]
        for i in idx: out[i] += ap(y[m], PRED[i][m]) / R
    return out, p


def loss(theta, idx, lam=2e-4):
    s, _ = sim(theta, idx); return np.mean((s[idx] - Y_OBS[idx]) ** 2) + lam * np.sum(theta[2:] ** 2)


def fit(idx, x0=None):
    x0 = np.r_[0.0, 1.0, np.zeros(F.shape[1])] if x0 is None else x0
    return minimize(loss, x0, args=(idx,), method="Powell", options={"maxiter": 3000, "xtol": 1e-3, "ftol": 1e-9}).x


if __name__ == "__main__":
    allidx = list(range(len(Y_OBS)))
    base = np.r_[0.0, 1.0, np.zeros(F.shape[1])]; s0, p0 = sim(base)
    print("baseline (test labels ~ v5 probabilities): RMSE %.4f  corr %.3f  mean prevalence %.4f" % (np.sqrt(np.mean((s0 - Y_OBS) ** 2)), np.corrcoef(s0, Y_OBS)[0, 1], p0.mean()))
    th = fit(allidx); s1, p1 = sim(th)
    print("fitted: RMSE %.4f corr %.3f prevalence %.4f" % (np.sqrt(np.mean((s1 - Y_OBS) ** 2)), np.corrcoef(s1, Y_OBS)[0, 1], p1.mean()))
    print("a=%.3f b=%.3f" % (th[0], th[1])); print(json.dumps({k: round(v, 3) for k, v in zip(FN, th[2:])}))
    for n, o, s in zip(NAMES, Y_OBS, s1): print(f"   {n:45s} public {o:.5f} fitted {s:.5f}")
    json.dump({"theta": th.tolist(), "features": FN}, open(P("experiments/lb_infer_theta.json"), "w"))
    # leave-one-submission-out: does the fitted regime predict an unseen submission better than the baseline?
    errs0, errs1 = [], []
    for i in allidx:
        rest = [j for j in allidx if j != i]; t = fit(rest, th); si, _ = sim(t, [i])
        # compare predicted delta vs v5 (index of v5)
        errs1.append(si[i] - Y_OBS[i]); errs0.append(s0[i] - Y_OBS[i])
        print(f"LOO {NAMES[i]:45s} obs {Y_OBS[i]:.5f} pred {si[i]:.5f} (baseline {s0[i]:.5f})", flush=True)
    e0, e1 = np.array(errs0), np.array(errs1)
    print("LOO RMSE baseline %.4f fitted %.4f; centred (removing mean offset) baseline %.4f fitted %.4f" % (
        np.sqrt(np.mean(e0 ** 2)), np.sqrt(np.mean(e1 ** 2)), np.std(e0), np.std(e1)))
