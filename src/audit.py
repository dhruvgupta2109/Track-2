"""Candidate audit: OOF views + test predictions + correlations -> experiments/ALL_CANDIDATES.csv"""
import sys, os, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score as ap
from evaluate import Y, W_HN, W_COV, TR, TE, prevalence_deltas
from candidates import REG, load_oof

PUBLIC = {"v1_lr_eng": 0.17617, "v3_lr": 0.17206, "v4_lr": 0.17966, "v5_lr_knn": 0.18626, "v4_lambda05_probe": 0.16822}
res = {}
for l in open("experiments/results_v2.jsonl"):
    r = json.loads(l); res[r["name"]] = r
alias = {"oof5_v4": "v4_prod", "oof5_v5_prod": "r7_combo_age_ctx_knn", "oof5_v5a_prod": "r7_combo_age_ctx", "oof5_v3_lr": "v3_lr", "oof5_v5b_burstnight": "v5b_burstnight",
         "oof5_v3+burst+nd": "v3+burst+nd", "oof5_catboost_d4": "catboost_d4", "oof5_lgb_mono": "lgb_mono", "oof5_r6_knn": "r6_knn_full",
         "oof5_r6_knn_dratio_full": "r6_knn_dratio_full", "oof5_v4_cross_nd_hour": "v4_cross_nd_hour"}
os.makedirs("submissions/audit", exist_ok=True)
O = {n: load_oof(v[0]) for n, v in REG.items()}
o5, o4 = O["v5_lr_knn"], O["v4_lr"]
T = {}
for n, (stem, fam, gen, dep, verdict) in REG.items():
    path = f"submissions/sub_{n}.csv" if os.path.exists(f"submissions/sub_{n}.csv") else f"submissions/audit/cand_{n}.csv"
    if n == "v5b_burstnight": path = "submissions/sub_v5b_burstnight_probe.csv"
    if os.path.exists(path): T[n] = pd.read_csv(path).prediction.values
    elif gen is not None:
        p = gen(TR, TE); pd.DataFrame({"id": TE.id, "prediction": np.clip(p, 0, 1)}).to_csv(path, index=False); T[n] = p
    REG[n] = (stem, fam, gen, dep, verdict, path if n in T else "")
    print("ready", n, flush=True)
rows = []
for n, (stem, fam, gen, dep, verdict, path) in REG.items():
    o = O[n]; R = range(len(o)); r = res.get(alias.get(stem, stem), res.get(stem.replace("oof5_", ""), {}))
    std = np.mean([ap(Y, o[i]) for i in R]); hn = np.mean([ap(Y, o[i], sample_weight=W_HN) for i in R])
    pv = prevalence_deltas(o, o5, prevs=(0.01,), n=60)[0.01]
    top = lambda x, k: np.mean([Y[np.argsort(-x[i])[:int(k * len(Y))]].sum() / Y.sum() for i in R])
    t = T.get(n)
    rows.append({"candidate_name": n, "prediction_file": path, "experiment_family": fam, "dependency_requirements": dep, "public": PUBLIC.get(n),
                 "standard_cv": round(std, 4), "second_seed_cv": r.get("seed2"), "hard_negative_cv": round(hn, 4), "missing_stress_cv": r.get("miss"),
                 "hn_missing_cv": r.get("hnW_miss"), "test_like_holdout": r.get("holdout"), "test_like_holdout_hn": r.get("holdout_hnW"),
                 "delta_vs_v4": round(std - np.mean([ap(Y, o4[i]) for i in range(5)]), 4), "delta_vs_v5": round(std - np.mean([ap(Y, o5[i]) for i in range(5)]), 4),
                 "hn_delta_vs_v5": round(hn - np.mean([ap(Y, o5[i], sample_weight=W_HN) for i in range(5)]), 4), "ap_delta_vs_v5_at_1pct_prev": round(pv, 4),
                 "recall_top1pct": round(top(o, 0.01), 3), "recall_top5pct": round(top(o, 0.05), 3),
                 "oof_rank_corr_v5": round(np.mean([spearmanr(o[i], o5[i])[0] for i in R]), 4),
                 "prediction_correlation_with_v5": round(np.corrcoef(t, T["v5_lr_knn"])[0, 1], 4) if t is not None else None,
                 "rank_correlation_with_v5": round(spearmanr(t, T["v5_lr_knn"])[0], 4) if t is not None else None,
                 "prediction_correlation_with_v4": round(np.corrcoef(t, T["v4_lr"])[0, 1], 4) if t is not None else None,
                 "rank_correlation_with_v4": round(spearmanr(t, T["v4_lr"])[0], 4) if t is not None else None,
                 "previous_verdict": verdict, "can_regenerate": gen is not None})
df = pd.DataFrame(rows).sort_values("standard_cv", ascending=False)
df.to_csv("experiments/ALL_CANDIDATES.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print(df[["candidate_name", "public", "standard_cv", "second_seed_cv", "hard_negative_cv", "missing_stress_cv", "test_like_holdout", "test_like_holdout_hn",
          "delta_vs_v5", "hn_delta_vs_v5", "recall_top1pct", "recall_top5pct", "oof_rank_corr_v5", "rank_correlation_with_v5"]].to_string(index=False))
# public-delta tracking (vs v3 as common baseline)
print("\nlocal vs public deltas (baseline v3):")
for n in ["v1_lr_eng", "v4_lr", "v5_lr_knn"]:
    d = df.set_index("candidate_name")
    print(f"  {n:12s} std {d.loc[n,'standard_cv']-d.loc['v3_lr','standard_cv']:+.4f}  hnW {d.loc[n,'hard_negative_cv']-d.loc['v3_lr','hard_negative_cv']:+.4f}  "
          f"public {PUBLIC[n]-PUBLIC['v3_lr']:+.5f}")
