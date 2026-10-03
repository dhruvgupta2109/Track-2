"""v5 design + the programs selected by L1 synthesis INSIDE the same outer fold (nested: selection never saw the validation fold).
L2 logistic regression; program block scaled by m (block-specific penalty)."""
import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "src")
from evaluate import TR, Y, SPLITS
from apopt_lab import design
from synth_lab import library
SEL = {s["fold"]: s["selected"] for s in json.load(open("experiments/synth_l1_selection.json"))}
CFG = [(0.3, 0.1), (0.5, 0.1), (1.0, 0.1), (0.5, 0.05)]
oofs = {c: np.full((5, len(Y)), np.nan) for c in CFG}
for i, (a, b) in enumerate(SPLITS):
    r = i // 5; A, B = TR.iloc[a], TR.iloc[b]
    XA1, XB1, ya, _, _ = design(A, B)
    cat_med = A.groupby("merchant_category").transaction_amount.median().to_dict()
    lv = {c: sorted(A[c].dropna().unique()) for c in ["merchant_category", "country", "transaction_channel"]}
    LA, LB = library(A, cat_med, lv)[SEL[i]], library(B, cat_med, lv)[SEL[i]]
    med = LA.median(); mu = LA.fillna(med).mean(); sd = LA.fillna(med).std().replace(0, 1)
    PA = ((LA.fillna(med) - mu) / sd).values; PB = ((LB.fillna(med) - mu) / sd).values
    for m, C in CFG:
        oofs[(m, C)][r, b] = LogisticRegression(C=C, max_iter=8000).fit(np.hstack([XA1, m * PA]), ya).predict_proba(np.hstack([XB1, m * PB]))[:, 1]
import arch_eval
for (m, C), o in oofs.items():
    np.save(f"experiments/oof5_v5prog_m{m}_C{C}.npy", o); arch_eval.evaluate(o, f"v5+fold-selected programs m{m} C{C}")
