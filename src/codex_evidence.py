"""Recompute judge-facing facts, fresh full-fit reproduction and selection gate."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from threadpoolctl import threadpool_limits

from build_v5 import fit_v5
from codex_delivery import ROOT, PACKAGE
from codex_polylab import fit_poly, fit_galr
from r2_runtime import predict_r2
from templates import template_masks


def evaluate_evidence():
    tr, te = pd.read_csv(ROOT / "train.csv"), pd.read_csv(ROOT / "test.csv")
    y = tr.fraud.to_numpy()
    original = joblib.load(PACKAGE / "model.pkl")
    with threadpool_limits(limits=2):
        # Fresh full training, using no cached OOFs and no public labels.
        rebuilt = fit_v5(tr)
        pm, pc = fit_poly(tr, "original", context=te)
        rebuilt.update(poly_model=pm, poly_constants=pc, blend_w_poly=0.3,
                       galr_model=fit_galr(tr), galr_columns=original["galr_columns"], blend_w_galr=0.2)
        p, details = predict_r2(te, original, return_details=True)
        fresh = predict_r2(te, rebuilt)
    reproduction_diff = float(np.max(np.abs(p - fresh)))
    if reproduction_diff >= 1e-9:
        raise AssertionError(f"Fresh full-training reproduction failed: {reproduction_diff}")
    pd.DataFrame({"id": te.id, "prediction": details["ranking"]}).to_csv(PACKAGE / "predictions_before_operating_shift.csv", index=False)
    results = pd.read_csv(ROOT / "experiments" / "codex" / "polylab_2718_811_1201.csv")
    confirmation = results[results.seed.isin([811, 1201])]
    gate = []
    for variant in ("loo_risk", "interaction_only", "capped_inputs"):
        s = confirmation[(confirmation.variant == variant) & (confirmation.view == "standard")]
        m = confirmation[(confirmation.variant == variant) & (confirmation.view == "extra_missing")]
        accepted = bool(s.delta_ap.mean() >= 0.001 and (s.delta_ap > 0).all() and m.delta_ap.mean() >= -0.001)
        gate.append(dict(variant=variant, confirmation_delta=float(s.delta_ap.mean()),
                         missingness_delta=float(m.delta_ap.mean()), both_seeds_positive=bool((s.delta_ap > 0).all()),
                         accepted=accepted))
    pd.DataFrame(gate).to_csv(ROOT / "experiments" / "codex" / "selection_gate.csv", index=False)
    Mtr, Mte = template_masks(tr), template_masks(te)
    clean = te.transaction_hour.isin([5, 6, 7, 21, 22])
    inferred_base = int(clean.sum()) * 24 / 5
    profile = []
    for name in Mtr:
        profile.append(dict(segment=name, train_rows=int(Mtr[name].sum()), train_frauds=int(y[Mtr[name]].sum()),
                            train_observed_rate=float(y[Mtr[name]].mean()), test_rows=int(Mte[name].sum()),
                            test_label_rate="unknown; not observed"))
    pd.DataFrame(profile).to_csv(ROOT / "experiments" / "codex" / "template_profile.csv", index=False)
    missing = pd.DataFrame({"train": tr.drop(columns=["id", "fraud"]).isna().mean(),
                            "test": te.drop(columns=["id"]).isna().mean()})
    missing.to_csv(ROOT / "experiments" / "codex" / "missingness.csv")
    flags = []
    for view in ("standard", "extra_missing"):
        for seed in (2718, 811, 1201):
            oof = np.load(ROOT / "experiments" / "codex" / f"oof_{seed}_{view}_original.npy")
            for fraction in (0.005, 0.01, 0.02, 0.025, 0.03, 0.05):
                count = int(round(len(y) * fraction))
                decision = np.zeros(len(y), dtype=int)
                decision[np.argsort(-oof, kind="stable")[:count]] = 1
                flags.append(dict(seed=seed, view=view, policy="top_fraction_within_labeled_train_OOF",
                                  flag_fraction=fraction, flagged=count, precision=precision_score(y, decision),
                                  recall=recall_score(y, decision), f1=f1_score(y, decision)))
    # These are labeled training-OOF workload curves, not hidden-test estimates,
    # and do NOT validate the package's fixed score threshold under concept shift.
    pd.DataFrame(flags).to_csv(ROOT / "experiments" / "codex" / "oof_workload_metrics.csv", index=False)
    point = float(expit(-2.188389))
    paired = {}
    baseline = np.mean([np.load(ROOT / "experiments" / "codex" / f"oof_{seed}_standard_original.npy") for seed in (811, 1201)], axis=0)
    alternative = np.mean([np.load(ROOT / "experiments" / "codex" / f"oof_{seed}_standard_interaction_only.npy") for seed in (811, 1201)], axis=0)
    # Paired stratified row bootstrap. It quantifies row uncertainty conditional on
    # the fitted OOF predictions; it is not full retraining uncertainty.
    rng = np.random.default_rng(8401)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    deltas = []
    for _ in range(400):
        ix = np.r_[rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        deltas.append(average_precision_score(y[ix], alternative[ix]) - average_precision_score(y[ix], baseline[ix]))
    paired["interaction_only"] = {"delta_on_confirmation_mean_oof": float(average_precision_score(y, alternative) - average_precision_score(y, baseline)),
                                  "conditional_row_bootstrap_ci95": np.percentile(deltas, [2.5, 97.5]).tolist(), "draws": 400}
    order_preserved = bool(np.array_equal(np.argsort(p, kind="stable"), np.argsort(details["ranking"], kind="stable")))
    if not order_preserved:
        raise AssertionError("Operating-point shift unexpectedly changed ranking")
    facts = {"train_rows": len(tr), "test_rows": len(te), "frauds": int(y.sum()), "train_prevalence": float(y.mean()),
             "train_burst_rows": int((tr.transactions_last_24h >= 16).sum()),
             "test_burst_rows": int((te.transactions_last_24h >= 16).sum()),
             "burst_rate_ratio": float((te.transactions_last_24h >= 16).mean() / (tr.transactions_last_24h >= 16).mean()),
             "train_mean_feature_missingness": float(missing.train.mean()), "test_mean_feature_missingness": float(missing.test.mean()),
             "hour_fingerprint_base_estimate": inferred_base, "hour_fingerprint_excess_estimate": len(te) - inferred_base,
             "hour_fingerprint_assumption": "Base hours uniform, no feature-dependent missing-hour bias; this is an estimate, not a row label.",
             "fresh_standard_base_ap_mean": float(results[(results.variant == "original") & (results.view == "standard")].base_ap.mean()),
             "fresh_missing_base_ap_mean": float(results[(results.variant == "original") & (results.view == "extra_missing")].base_ap.mean()),
             "fresh_full_fit_max_difference": reproduction_diff, "gate": gate, "paired_bootstrap": paired,
             "operating_shift": 2.188389, "equivalent_pre_shift_threshold": point,
             "flagged_before_operating_shift": int((details["ranking"] >= 0.5).sum()), "flagged_after_operating_shift": int((p >= 0.5).sum()),
             "operating_shift_preserves_test_ranking": order_preserved, "new_public_score": None,
             "unknowns": ["private score", "private evaluator decision threshold", "test labels", "true T1/T2/T3 rates", "exact data generator", "achievable PR-AUC ceiling"]}
    (ROOT / "experiments" / "codex" / "evidence.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(json.dumps(facts, indent=2))
    return facts


if __name__ == "__main__":
    evaluate_evidence()
