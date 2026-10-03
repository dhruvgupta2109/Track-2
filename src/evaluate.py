"""Multi-view validation for Track 2.

Views (all label-leak-free; test-likeness comes from a train-vs-test classifier that never sees fraud labels):
  std      : 5x5 RepeatedStratifiedKFold (seed 2026) - mean / median / sd / min / max fold AP
  seed2    : 5x3 RSKF with a different seed (stability)
  covW     : AP with covariate-shift importance weights on ALL rows (assumes P(y|x) unchanged) - proved over-optimistic
  hnW      : "hard-negative" view: only LEGIT rows re-weighted to test covariates (test = train frauds + test-like legit
             population). Reproduced the first public score (0.180 vs 0.172), so it is the main shift-aware view.
  hnW_miss : hnW on OOF predictions made with test-level missingness injected into validation folds
  holdout  : train on the 80% least test-like rows, evaluate on the 20% most test-like (plain AP and hnW AP)
  boot     : bootstrap 90% CI of the hnW and std AP difference vs a reference OOF (paired)
"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import load, cv_splits, run_cv, LABEL, ROOT
from sklearn.metrics import average_precision_score as ap, roc_auc_score

TR, TE = load()
Y = TR[LABEL].values
_tl = np.load(os.path.join(ROOT, "experiments", "testlikeness_train.npy"))
_r = np.clip(_tl, 0.01, 0.99); _w = np.minimum(_r / (1 - _r), 10.0)
W_COV = _w / _w.mean()
W_HN = np.where(Y == 0, _w / _w[Y == 0].mean(), 1.0)
HOLD = _tl >= np.quantile(_tl, 0.8)
SPLITS = cv_splits(Y, n_repeats=5, seed=2026)
SPLITS2 = cv_splits(Y, n_repeats=3, seed=11)


def views_from_oof(oof, splits=SPLITS):
    k = len(splits) // len(oof)
    fold = [ap(Y[b], oof[i // k, b]) for i, (a, b) in enumerate(splits)]
    R = range(len(oof))
    return {"std": np.mean([ap(Y, oof[r]) for r in R]), "std_median_fold": np.median(fold), "std_sd_fold": np.std(fold),
            "std_min_fold": np.min(fold), "std_max_fold": np.max(fold),
            "covW": np.mean([ap(Y, oof[r], sample_weight=W_COV) for r in R]),
            "hnW": np.mean([ap(Y, oof[r], sample_weight=W_HN) for r in R]),
            "roc": np.mean([roc_auc_score(Y, oof[r]) for r in R]),
            "hn_recall@top2%": np.mean([_recall_top(oof[r], 0.02, W_HN) for r in R])}


def _recall_top(p, frac, w):
    order = np.argsort(-p); cw = np.cumsum(w[order]); cut = order[: np.searchsorted(cw, frac * w.sum()) + 1]
    return Y[cut].sum() / Y.sum()


def boot_diff(p_new, p_ref, w=None, n=400, seed=0):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.choice(len(Y), len(Y), replace=True); ww = None if w is None else w[i]
        d.append(ap(Y[i], p_new[i], sample_weight=ww) - ap(Y[i], p_ref[i], sample_weight=ww))
    return np.percentile(d, [5, 50, 95]), np.mean(np.array(d) > 0)


def evaluate(name, fp, quick=False, ref="v3_lr", save=True):
    """fp(train_df, valid_df) -> fraud probabilities for valid_df."""
    oof, _ = run_cv(fp, TR, SPLITS, 5, verbose=False)
    v = views_from_oof(oof)
    if not quick:
        o2, _ = run_cv(fp, TR, SPLITS2, 3, verbose=False); v["seed2"] = np.mean([ap(Y, o2[r]) for r in range(3)])
        om, _ = run_cv(fp, TR, SPLITS[:10], 2, extra_missing=0.017, verbose=False)
        v["miss"] = np.mean([ap(Y, om[r]) for r in range(2)]); v["hnW_miss"] = np.mean([ap(Y, om[r], sample_weight=W_HN) for r in range(2)])
        ph = fp(TR[~HOLD], TR[HOLD]); v["holdout"] = ap(Y[HOLD], ph); v["holdout_hnW"] = ap(Y[HOLD], ph, sample_weight=W_HN[HOLD])
    refp = os.path.join(ROOT, "experiments", f"oof5_{ref}.npy")
    if os.path.exists(refp) and name != ref:
        r = np.load(refp).mean(0); m = oof.mean(0)
        (lo, med, hi), pw = boot_diff(m, r); v["d_std_ci90"] = f"{lo:+.4f}..{hi:+.4f}"
        (lo, med, hi), pw2 = boot_diff(m, r, W_HN); v["d_hnW_ci90"] = f"{lo:+.4f}..{hi:+.4f}"; v["P(d_hnW>0)"] = pw2
    if save:
        np.save(os.path.join(ROOT, "experiments", f"oof5_{name}.npy"), oof)
        with open(os.path.join(ROOT, "experiments", "results_v2.jsonl"), "a") as f:
            f.write(json.dumps({"name": name, **{k: (round(float(x), 5) if not isinstance(x, str) else x) for k, x in v.items()}}) + "\n")
    print(name.ljust(28), " ".join(f"{k}={x:.4f}" if not isinstance(x, str) else f"{k}={x}" for k, x in v.items()), flush=True)
    return oof, v
