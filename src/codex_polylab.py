"""Small, predeclared interaction-LR robustness experiment against the R2 base.

No public scores are fitted. Supervised constants and neighbour references are
learned inside each outer fold. As in the shipped interaction member, category
amount medians use train plus unlabeled evaluation features (transductive).
Extra missingness affects validation rows only.
The screen seed selects hypotheses; two new seeds confirm them. Local gains
never constitute a new public score or justify replacing the public winner.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

from build_v5 import fit_v5, predict_v5
from common import RAW, inject_missing
from constants import fit_constants
from features_prod import sanitize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "codex"
CAT = ["merchant_category", "country", "transaction_channel"]
CORE = ["log_amt", "log_spend", "log_age", "transactions_last_24h",
        "transactions_last_1h", "new_device", "amt_vs_cat", "foreign",
        "young", "burst", "night"]
NUM = CORE + [c + "_risk" for c in CAT]
VARIANTS = ("original", "loo_risk", "interaction_only", "capped_inputs")
CONT = CORE[:5] + ["amt_vs_cat"]


def constants(a, context=None):
    y = a.fraud.to_numpy()
    prior = float(y.mean())
    risk = {}
    for c in CAT:
        g = a.groupby(a[c].fillna("NA")).fraud.agg(["sum", "count"])
        p = (g["sum"] + 20 * prior) / (g["count"] + 20)
        risk[c] = logit(p).to_dict()
    # Auditing the saved Q30 member revealed log1p(median(raw amount)) on
    # train+public-test features, not median(log amount) on training alone.
    # Fraud labels are never read from the optional context.
    med_source = a if context is None else pd.concat([a[RAW], context[RAW]], ignore_index=True)
    return {"cat_med_log": np.log1p(med_source.groupby("merchant_category").transaction_amount.median()).to_dict(),
            "risk_maps": risk, "risk_prior": float(logit(prior)),
            "cat": CAT, "core": CORE}


def poly_frame(df, pc):
    d = pd.DataFrame(index=df.index)
    for raw, name in (("transaction_amount", "log_amt"), ("spend_last_24h", "log_spend"), ("account_age", "log_age")):
        d[name] = np.log1p(df[raw])
    for c in ("transactions_last_24h", "transactions_last_1h", "new_device"):
        d[c] = df[c]
    d["amt_vs_cat"] = d.log_amt - df.merchant_category.map(pc["cat_med_log"]).astype(float)
    d["foreign"] = (df.country != "SG").astype(float).where(df.country.notna())
    d["young"] = (df.account_age < 180).astype(float).where(df.account_age.notna())
    d["burst"] = (df.transactions_last_24h >= 16).astype(float).where(df.transactions_last_24h.notna())
    d["night"] = df.transaction_hour.isin([0, 1, 2, 3, 4, 22, 23]).astype(float)
    for c in CAT:
        d[c] = df[c]
        d[c + "_risk"] = df[c].fillna("NA").map(pc["risk_maps"][c]).astype(float).fillna(pc["risk_prior"])
    for c, bounds in pc.get("caps", {}).items():
        d[c] = d[c].clip(*bounds)
    return d


def poly_pipe(interaction_only=False):
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            PolynomialFeatures(2, include_bias=False, interaction_only=interaction_only), StandardScaler())
    categorical = make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(handle_unknown="ignore"))
    return make_pipeline(ColumnTransformer([("num", numeric, NUM), ("cat", categorical, CAT)]),
                         LogisticRegression(C=0.003, max_iter=5000))


def fit_poly(a, variant, context=None):
    pc = constants(a, context)
    A = poly_frame(a, pc)
    if variant == "loo_risk":
        y = a.fraud.to_numpy()
        prior_without_self = (y.sum() - y) / (len(y) - 1)
        for c in CAT:
            keys = a[c].fillna("NA")
            g = a.groupby(keys).fraud.agg(["sum", "count"])
            p = (keys.map(g["sum"]).to_numpy() - y + 20 * prior_without_self) / (keys.map(g["count"]).to_numpy() - 1 + 20)
            A[c + "_risk"] = logit(np.clip(p, 1e-9, 1 - 1e-9))
    elif variant == "capped_inputs":
        pc["caps"] = {c: (float(A[c].mean() - 3 * A[c].std(ddof=0)), float(A[c].mean() + 3 * A[c].std(ddof=0))) for c in CONT}
        A = poly_frame(a, pc)
    model = poly_pipe(variant == "interaction_only").fit(A, a.fraud.to_numpy())
    return model, pc


def galr_frame(df):
    d = pd.DataFrame(index=df.index)
    for t in [100, 200, 300, 500, 750, 1000, 1500, 2500]:
        d[f"amt_gt{t}"] = (df.transaction_amount > t).astype(float).where(df.transaction_amount.notna())
    d["night"] = df.transaction_hour.isin([23, 0, 1, 2, 3, 4]).astype(float).where(df.transaction_hour.notna())
    for t in [120, 270, 365]:
        d[f"age_lt{t}"] = (df.account_age < t).astype(float).where(df.account_age.notna())
    d["t1_pos"] = (df.transactions_last_1h > 0).astype(float).where(df.transactions_last_1h.notna())
    d["t24_clip"] = df.transactions_last_24h.clip(upper=15)
    d["t24_excess"] = (df.transactions_last_24h - 15).clip(lower=0)
    d["new_device"] = df.new_device
    d["nd_x_night"] = df.new_device * d.night
    d["nd_x_ecom"] = df.new_device * (df.transaction_channel == "ecommerce").astype(float).where(df.transaction_channel.notna())
    for c in CAT:
        d[c] = df[c]
    d["hour_c"] = df.transaction_hour.round().astype("Int64").astype(str).where(df.transaction_hour.notna(), None)
    return d


def fit_galr(a):
    A = galr_frame(a)
    cats = CAT + ["hour_c"]
    num = [c for c in A if c not in cats]
    ct = ColumnTransformer([
        ("c", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"), OneHotEncoder(handle_unknown="ignore")), cats),
        ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ])
    return make_pipeline(ct, LogisticRegression(C=0.05, max_iter=5000)).fit(A, a.fraud.to_numpy())


def blend(v5, poly, galr):
    lg = lambda p: logit(np.clip(p, 1e-9, 1 - 1e-9))
    return expit(0.56 * lg(v5) + 0.24 * lg(poly) + 0.20 * lg(galr))


def run(seed):
    tr = pd.read_csv(ROOT / "train.csv")
    y = tr.fraud.to_numpy()
    views = ("standard", "extra_missing")
    scores = {v: {k: np.zeros(len(tr)) for k in VARIANTS} for v in views}
    member = {v: {k: np.zeros(len(tr)) for k in VARIANTS} for v in views}
    started = time.monotonic()
    for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y), 1):
        A, B = tr.iloc[a], tr.iloc[b]
        K = fit_constants(A)
        # Match the package's sanitised inference, including clipping and rounding.
        Bviews = {"standard": sanitize(B, K), "extra_missing": sanitize(inject_missing(B, 0.017, np.random.default_rng(seed * 10 + fold)), K)}
        v5 = fit_v5(A)
        gm = fit_galr(A)
        polys = {k: fit_poly(A, k, context=B[RAW]) for k in VARIANTS}
        for view, V in Bviews.items():
            pv = predict_v5(v5, V)
            pg = gm.predict_proba(galr_frame(V))[:, 1]
            for name, (pm, pc) in polys.items():
                pp = pm.predict_proba(poly_frame(V, pc))[:, 1]
                scores[view][name][b] = blend(pv, pp, pg)
                member[view][name][b] = pp
        print(f"seed={seed} fold={fold}/5 elapsed={time.monotonic()-started:.1f}s", flush=True)
    rows = []
    for view in views:
        ref = average_precision_score(y, scores[view]["original"])
        for name in VARIANTS:
            p = scores[view][name]
            ap = average_precision_score(y, p)
            rows.append(dict(seed=seed, view=view, variant=name, base_ap=ap, delta_ap=ap-ref,
                             member_ap=average_precision_score(y, member[view][name]), brier=brier_score_loss(y, p)))
            np.save(OUT / f"oof_{seed}_{view}_{name}.npy", p)
            print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[2718])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    rows = []
    with threadpool_limits(limits=2):
        for seed in args.seeds:
            rows.extend(run(seed))
    pd.DataFrame(rows).to_csv(OUT / ("polylab_" + "_".join(map(str, args.seeds)) + ".csv"), index=False)
    meta = {"seeds": args.seeds, "variants": VARIANTS, "data_sha256": hashlib.sha256((ROOT / "train.csv").read_bytes()).hexdigest(),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "selection": "Screen seed 2718; confirmation seeds 811 and 1201. No hyperparameter search. Transductive category medians use outer-training plus unlabeled validation features, matching the saved-member recipe; fraud-risk maps use outer-training labels only.",
            "gate": "Mean confirmation base-AP delta >=0.001; both standard seed deltas positive; mean extra-missing delta >=-0.001. Prepare optional candidate only; public winner remains R2op."}
    (OUT / ("polylab_protocol_" + "_".join(map(str, args.seeds)) + ".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
