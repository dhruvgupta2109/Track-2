"""Fresh-seed check + nested CV for 'Q30 base (0.7 v5 + 0.3 interaction LR) + w * GALR'.
Fresh seeds: all three base models refit on new shuffled 5-fold splits (seeds 11, 99, 314).
Nested: per outer fold, (C, w) chosen by inner 4-fold CV on the outer-train part only; evaluated on the outer-test part."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, "src")
from build_v5 import fit_v5, predict_v5
from blendlab import fp_poly
from galr import fp as fp_galr
tr = pd.read_csv("train.csv"); y = tr.fraud.values
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
CS, WS = (0.03, 0.05, 0.1), (0.0, 0.1, 0.2, 0.3)


def base_preds(a, b):
    A, B = tr.iloc[a], tr.iloc[b]
    out = {"v5": predict_v5(fit_v5(A), B), "poly": fp_poly()(A, B)}
    for C in CS: out[f"g{C}"] = fp_galr(C)(A, B)
    return out


def combo(P, C, w): return (1 - w) * (0.7 * lg(P["v5"]) + 0.3 * lg(P["poly"])) + w * lg(P[f"g{C}"])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fresh"
    if mode == "fresh":
        for seed in (11, 99, 314):
            oof = {}
            for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y):
                P = base_preds(a, b)
                for k, v in P.items(): oof.setdefault(k, np.zeros(len(y)))[b] = v
            base = ap(y, combo(oof, 0.05, 0.0))
            print(f"seed {seed}: v5 {ap(y, oof['v5']):.4f} | Q30 base {base:.4f} | GALR alone " + " ".join(f"C{C} {ap(y, oof[f'g{C}']):.4f}" for C in CS) +
                  " | blend deltas " + " ".join(f"C{C}/w{w} {ap(y, combo(oof, C, w)) - base:+.4f}" for C in CS for w in WS[1:]), flush=True)
    else:
        for seed in (2027, 4242):
            outer_best, outer_sel, outer_chosen = np.zeros(len(y)), np.zeros(len(y)), []
            for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y):
                inner = {}
                for ia, ib in StratifiedKFold(4, shuffle=True, random_state=seed + 1).split(a, y[a]):
                    P = base_preds(a[ia], a[ib])
                    for k, v in P.items(): inner.setdefault(k, np.zeros(len(a)))[ib] = v
                C, w = max(((C, w) for C in CS for w in WS), key=lambda cw: ap(y[a], combo(inner, *cw)))
                P = base_preds(a, b); outer_sel[b] = combo(P, C, w); outer_best[b] = combo(P, 0.05, 0.0); outer_chosen.append((C, w))
            print(f"nested seed {seed}: Q30 base {ap(y, outer_best):.4f} | nested-selected (C,w) {ap(y, outer_sel):.4f} "
                  f"({ap(y, outer_sel) - ap(y, outer_best):+.4f}); chosen per outer fold {outer_chosen}", flush=True)
