"""Single full fit vs ensembles of shuffled 5-fold runs (fold models' test predictions averaged: probabilities or ranks), GALR C=0.05.
Evaluated inside an outer 5-fold CV (2 seeds) so the comparison is out-of-sample."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as ap
from scipy.stats import rankdata
sys.path.insert(0, "src")
from galr import fp
tr = pd.read_csv("train.csv"); y = tr.fraud.values; f = fp(0.05)
for seed in (7, 8):
    full, bag_p, bag_r = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y):
        A, B = tr.iloc[a], tr.iloc[b]; full[b] = f(A, B); ps = []
        for run in range(6):
            for ia, _ in StratifiedKFold(5, shuffle=True, random_state=100 * seed + run).split(A, y[a]): ps.append(f(A.iloc[ia], B))
        ps = np.array(ps); bag_p[b] = ps.mean(0); bag_r[b] = np.mean([rankdata(p) / len(p) for p in ps], 0)
    # rank-averaged scores are only comparable within a fold; AP per fold then averaged for all three variants
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y))
    per = lambda s: np.mean([ap(y[b], s[b]) for _, b in folds])
    print(f"seed {seed}: full-fit AP {ap(y, full):.4f} / per-fold {per(full):.4f} | prob-avg 30 fold models {ap(y, bag_p):.4f} / {per(bag_p):.4f} | rank-avg per-fold {per(bag_r):.4f}", flush=True)
