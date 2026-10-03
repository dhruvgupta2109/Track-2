"""Joint logistic regression on the union of the three successful feature sets (v5 design incl. kNN ratio, GALR step basis,
pairwise-interaction expansion), instead of averaging three separately fitted models. Block multipliers m_v5, m_galr, m_poly scale
each standardised block (= block-specific L2 strength). Same folds; OOF -> arch_eval."""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, "src")
from evaluate import TR, Y, SPLITS
from apopt_lab import design
import galr
from blendlab import poly_consts, poly_frame, poly_pipe


def blocks(a, b):
    XA1, XB1, ya, _, _ = design(a, b)
    GA_, GB_ = galr.basis(a), galr.basis(b); num = [c for c in GA_.columns if c not in galr.CAT]
    ct = galr.pipe(num).steps[0][1].fit(GA_); XA2, XB2 = ct.transform(GA_), ct.transform(GB_)
    PC = poly_consts(a); PA, PB = poly_frame(a, PC), poly_frame(b, PC); ctp = poly_pipe(0.003, 2).steps[0][1].fit(PA)
    XA3, XB3 = ctp.transform(PA), ctp.transform(PB)
    d = lambda X: X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return [d(XA1), d(XA2), d(XA3)], [d(XB1), d(XB2), d(XB3)], ya


def fit_predict(BA, BB, ya, mult, C):
    XA = np.hstack([m * X for m, X in zip(mult, BA)]); XB = np.hstack([m * X for m, X in zip(mult, BB)])
    return LogisticRegression(C=C, max_iter=8000).fit(XA, ya).predict_proba(XB)[:, 1]


CFGS = {"j_1_1_1_C0.03": ((1, 1, 1), 0.03), "j_1_1_1_C0.1": ((1, 1, 1), 0.1), "j_1_.5_.3_C0.1": ((1, 0.5, 0.3), 0.1),
        "j_1_.5_.2_C0.1": ((1, 0.5, 0.2), 0.1), "j_1_.7_.5_C0.05": ((1, 0.7, 0.5), 0.05)}

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    oofs = {k: (np.load(f"experiments/oof5_{k}.npy") if os.path.exists(f"experiments/oof5_{k}.npy") else np.full((5, len(Y)), np.nan)) for k in CFGS}
    for i, (a, b) in enumerate(SPLITS[:reps * 5]):
        r = i // 5
        if all(not np.isnan(o[r, b]).any() for o in oofs.values()): continue
        BA, BB, ya = blocks(TR.iloc[a], TR.iloc[b])
        for k, (mult, C) in CFGS.items():
            oofs[k][r, b] = fit_predict(BA, BB, ya, mult, C); np.save(f"experiments/oof5_{k}.npy", oofs[k])
        print(f"fold {i} done", flush=True)
    import arch_eval
    for k in CFGS: arch_eval.evaluate(oofs[k], k)
