"""Build the separate, runnable judge showcase (not the sandbox upload)."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s))
    md('''# TrustGuard: finding fraud without disrupting ordinary customers
### Cookie Squad | NTU Datathon 2026 | Track 2

**Recorded public PR-AUC: 0.23434.** Our main gain came from recognising distribution shift and adapting a restrained logistic ensemble. This is the judge-facing analysis notebook. Upload `packages/codex_r2op_reviewed/prediction_notebook.ipynb` to the prediction card; the portal explicitly forbids training cells there.

This notebook runs local training, fresh validation, score reproduction and visual analysis using the supplied data. It distinguishes **observed data**, **historical public results**, and **unverified hypotheses**. No test labels, private scores or current leaderboard position are available.

**Demo route:** data imbalance -> train/test differences -> three complementary LR models -> validation -> exact prediction reproduction -> limitations and Q&A. A complete rerun uses CPU only and normally takes under a minute on this workspace.''')
    code('''from pathlib import Path
import sys, json, io, contextlib
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.special import expit
from sklearn.metrics import average_precision_score, precision_recall_curve
from threadpoolctl import threadpool_limits

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "train.csv").exists() and (p / "src" / "r2_runtime.py").exists())
sys.path.insert(0, str(ROOT / "src"))
from build_v5 import fit_v5
from codex_polylab import fit_poly, fit_galr, run
from r2_runtime import predict_r2, make_features, add_v5_features
from templates import template_masks

PACKAGE = ROOT / "packages" / "codex_r2op_reviewed"
FIGURES = ROOT / "analysis" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
tr, te = pd.read_csv(ROOT / "train.csv"), pd.read_csv(ROOT / "test.csv")
y = tr.fraud.to_numpy()
bundle = joblib.load(PACKAGE / "model.pkl")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 10, "axes.titlesize": 12, "font.size": 10})
NAVY, TEAL, RED = "#12283E", "#087F8C", "#CF5C54"
print(f"Data loaded: {len(tr):,} train, {len(te):,} test; {y.sum()} observed frauds. Test labels are unavailable.")''')
    md('''## 1. A useful model must find rare frauds

Only 353 of 20,000 training transactions are fraud. All-legitimate predictions achieve 98.235% accuracy while finding no fraud. Precision-recall ranking is the primary comparison; at a chosen review budget, precision measures analyst efficiency and recall measures fraud coverage.

The supplied data has no customer/entity or timestamp key for grouped or temporal validation. Stratified row folds are a practical limitation, not evidence of future customer-level generalisation.''')
    code('''assert set(tr.fraud.unique()) == {0, 1}
raw = [c for c in te if c != "id"]
facts = pd.DataFrame({"train": [len(tr), int(y.sum()), float(y.mean()), tr[raw].isna().mean().mean()],
                      "test": [len(te), "unavailable", "unavailable", te[raw].isna().mean().mean()]},
                     index=["rows", "observed frauds", "fraud prevalence", "mean feature missingness"])
display(facts)
print("Duplicate feature rows:", int(tr[raw].duplicated().sum()), "train;", int(te[raw].duplicated().sum()), "test")
overlap = tr[raw].merge(te[raw], on=raw, how="inner")
print("Exact feature overlap between train and test:", len(overlap))
fig, ax = plt.subplots(figsize=(8.2, 2.7))
ax.barh(["Legitimate", "Fraud"], [int((y==0).sum()), int(y.sum())], color=[TEAL, RED])
ax.set_xlabel("Labeled training transactions")
ax.set_title("353 frauds: model stability matters more than complexity", loc="left", color=NAVY)
for i, count in enumerate([int((y==0).sum()), int(y.sum())]): ax.text(count + 150, i, f"{count:,}", va="center")
ax.set_xlim(0, 22000)
fig.tight_layout(); fig.savefig(FIGURES / "01_class_imbalance.png", bbox_inches="tight"); plt.show()''')
    md('''## 2. The public test population differs from training

Compare **rates**, not raw row counts: the test is smaller. Transactions with at least 16 transactions in 24 hours occur 3.12 times as often in test. Their covariate profile resembles legitimate training bursts more than fraudulent training bursts; this is evidence for a population change, not observed test fraud labels.

Hour and feature support also suggest synthetic additions. Under a uniform base-hour assumption, hours 5, 6, 7, 21 and 22 imply about 8,962 base-like rows and 3,038 excess rows. This estimate depends on the assumption and missing-hour handling. It does not identify every added row or prove its label.''')
    code('''burst_tr = (tr.transactions_last_24h >= 16).to_numpy()
burst_te = (te.transactions_last_24h >= 16).to_numpy()
print(f"Burst rates: train {burst_tr.mean():.3%}; test {burst_te.mean():.3%}; ratio {burst_te.mean()/burst_tr.mean():.3f}x")
def burst_profile(df):
    return {"rows": len(df), "median account age": df.account_age.median(),
            "night share (23-4)": df.transaction_hour.isin([23,0,1,2,3,4]).mean(),
            "new-device share": (df.new_device==1).mean(), "median amount": df.transaction_amount.median()}
display(pd.DataFrame({"train legitimate bursts": burst_profile(tr[burst_tr & (y==0)]),
                      "train fraud bursts": burst_profile(tr[burst_tr & (y==1)]),
                      "unlabeled test bursts": burst_profile(te[burst_te])}).T.round(3))
missing = pd.DataFrame({"train": tr[raw].isna().mean(), "test": te[raw].isna().mean()})
hours = pd.DataFrame({"train": tr.transaction_hour.value_counts(normalize=True).reindex(range(24), fill_value=0),
                      "test": te.transaction_hour.value_counts(normalize=True).reindex(range(24), fill_value=0)})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
hours.plot.bar(ax=axes[0], color=[TEAL, RED], width=.8)
axes[0].set_title("Hour distribution (nonmissing observations)", loc="left"); axes[0].set_xlabel("Transaction hour"); axes[0].set_ylabel("Share")
(100*missing).plot.barh(ax=axes[1], color=[TEAL, RED])
axes[1].set_title("More missing feature values in test", loc="left"); axes[1].set_xlabel("Missing values (%)")
fig.tight_layout(); fig.savefig(FIGURES / "02_distribution_shift.png", bbox_inches="tight"); plt.show()
Mtr, Mte = template_masks(tr), template_masks(te)
display(pd.DataFrame([{"box": name, "train rows": int(Mtr[name].sum()), "train frauds": int(y[Mtr[name]].sum()),
                       "test rows": int(Mte[name].sum()), "test fraud rate": "unknown"} for name in Mtr]))''')
    md('''### Observations versus hypotheses

The T1/T2/T3 boxes are feature-defined masks that include some ordinary base rows. They are not ground-truth injected-row indicators. The final model changes only T2, shrinking its scores toward an indirectly inferred 3.5% working level. Neither exact test rates nor random labels within boxes are established. T3 was not probed; we do not invent its fraud rate.

The burst and new-device adjustments have distributional support. High-value promotions and the exact adjustment sizes depend more heavily on public-board selection. Unsupervised prior-shift estimators were tried previously but contradicted observed public directions; they were not adopted.''')
    md('''## 3. Model construction and fresh training reproduction

**v5:** regularised LR on engineered features and the log distance ratio to nearest fraud/legitimate references. A training row is excluded from its own nearest-neighbour match. At inference, stored training references and constants make each score row-local.

**Interaction LR:** degree-two products of 14 numeric features, plus categorical one-hot encoding; C=0.003. Smoothed category fraud-risk maps use labeled training rows. The saved member's amount medians use **log1p(median(raw amount)) from train plus unlabeled public-test features**. This transductive detail was recovered by auditing the stored constants and confirmed by reproduction. Reusing a row's label in its training category-risk feature is a self-influence concern; outer validation labels are excluded from those risk maps.

**GALR:** regularised LR on amount/age thresholds, hourly categories and limited device/activity interactions; C=0.05. It is weaker alone, but provides a modest complementary member.

The logit blend is **0.56 v5 + 0.24 interaction LR + 0.20 GALR**. Fixed group shifts and T2 shrinkage follow. We retain the original trained artifact. The next cell trains all three members from scratch without loading cached predictions and verifies the reproduced output.''')
    code('''with threadpool_limits(limits=2):
    rebuilt = fit_v5(tr)
    poly_model, poly_constants = fit_poly(tr, "original", context=te)
    rebuilt.update(poly_model=poly_model, poly_constants=poly_constants, blend_w_poly=.3,
                   galr_model=fit_galr(tr), galr_columns=bundle["galr_columns"], blend_w_galr=.2)
    original_p, details = predict_r2(te, bundle, return_details=True)
    reproduced_p = predict_r2(te, rebuilt)
maximum_difference = float(np.max(np.abs(original_p - reproduced_p)))
assert maximum_difference < 1e-9
print(f"Fresh full-training reproduction: maximum absolute score difference {maximum_difference:.3e}")
features = add_v5_features(make_features(te.head(5), bundle["constants"]), bundle["constants"])
display(features[["log_amt", "amt_rel_cat", "log_age", "age_lt120", "burst", "t1_ge4_nb", "nd_x_logage", "d_ratio"]].round(3))
print("Numerical constants, class order and model inputs are fixed at inference; IDs are never model features.")''')
    md('''## 4. Fresh validation and an explicit stopping rule

We tested three focused alternatives: leave-one-out category-risk encoding, interaction-only degree-two expansion (no squares), and caps at three training standard deviations on continuous interaction inputs. We kept C and blend weights fixed.

Seed 2718 was the screen; seeds 811 and 1201 were the confirmation. Five folds per seed exclude validation labels from training and fraud-risk mappings. Like the saved model's recipe, only **unlabeled validation features** are included in category medians. Extra-missingness validation independently blanks an additional 1.7% per feature. The results below measure the **base blend before public-test-specific group adjustments**, so they do not establish a private-test gain.

The stopping rule required a mean confirmation AP gain of at least 0.001, positive changes on both standard confirmation seeds and no mean missingness loss worse than 0.001. No variant passed. This is a predeclared decision rule, not a statistical significance threshold. Fresh seeds reuse the same underlying rows and labels; they are not independent new datasets.

This reruns the experiment so the notebook is reproducible after cloning, even though generated NumPy caches are not committed.''')
    code('''records = []
with threadpool_limits(limits=2), contextlib.redirect_stdout(io.StringIO()):
    for seed in (2718, 811, 1201): records.extend(run(seed))
results = pd.DataFrame(records)
display(results.pivot_table(index=["variant", "view"], columns="seed", values="base_ap").round(5))
confirmation = results[results.seed.isin([811,1201])]
gate = []
for name in ("loo_risk", "interaction_only", "capped_inputs"):
    standard = confirmation[(confirmation.variant==name) & (confirmation.view=="standard")]
    missing_view = confirmation[(confirmation.variant==name) & (confirmation.view=="extra_missing")]
    accepted = bool(standard.delta_ap.mean() >= .001 and (standard.delta_ap > 0).all() and missing_view.delta_ap.mean() >= -.001)
    gate.append({"variant":name, "mean confirmation AP change":standard.delta_ap.mean(),
                 "mean missingness AP change":missing_view.delta_ap.mean(), "accepted":accepted})
display(pd.DataFrame(gate).round(6))
assert not any(row["accepted"] for row in gate)
print("Decision: preserve R2op; no new public score is claimed.")
evidence = json.loads((ROOT / "experiments" / "codex" / "evidence.json").read_text())
display(pd.DataFrame(evidence["paired_bootstrap"]).T)
print("The paired bootstrap is conditional on fitted OOF scores; its interval is not retraining uncertainty.")''')
    code('''oof = np.mean([np.load(ROOT / "experiments" / "codex" / f"oof_{seed}_standard_original.npy") for seed in (2718,811,1201)], axis=0)
precision, recall, _ = precision_recall_curve(y, oof)
workload = pd.read_csv(ROOT / "experiments" / "codex" / "oof_workload_metrics.csv")
curve = workload[workload.view=="standard"].groupby("flag_fraction")[["precision", "recall", "f1"]].mean()
fig, axes = plt.subplots(1,2,figsize=(11,3.5))
axes[0].plot(recall, precision, color=TEAL, label=f"Mean fresh OOF scores: AP {average_precision_score(y,oof):.3f}")
axes[0].axhline(y.mean(), color=RED, linestyle="--", label=f"Training prevalence {y.mean():.3%}")
axes[0].set(xlabel="Recall", ylabel="Precision", xlim=(0,1), ylim=(0,1), title="Training-distribution precision-recall")
axes[0].legend(fontsize=8)
for name,color in zip(["precision","recall","f1"],[TEAL,RED,NAVY]): axes[1].plot(100*curve.index,curve[name], marker="o",label=name,color=color)
axes[1].set(xlabel="Review budget (% of labeled OOF rows)", ylabel="Metric", title="Workload versus fraud coverage")
axes[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(FIGURES / "03_validation_and_workload.png",bbox_inches="tight");plt.show()
display(curve.round(3))''')
    md('''The PR curve averages fresh OOF scores for visualisation; the comparison table averages each seed's AP. Those calculations need not have the same AP. Workload curves use labeled training OOF ranks and choose the top fraction in that dataset. They are **not private-test estimates**, nor validation of the package's fixed threshold under distribution shift.''')
    md('''## 5. Public history and final-score interpretation

The next table is a selected historical record copied from `experiments/LEADERBOARD.md`. It omits many failed probes and is not an independent validation set. Differences may reflect both model changes and selection on the same public labels. We have not fetched a new leaderboard score or verified today's rank.''')
    code('''history = pd.DataFrame({"submission": ["v3","v4","v5","S3","P1","Z3","Q30","R2op","R3op"],
                        "recorded public PR-AUC":[.17206,.17966,.18626,.20453,.21289,.22215,.23288,.23434,.23260]})
display(history)
fig,ax=plt.subplots(figsize=(9,2.8))
ax.plot(history.submission,history["recorded public PR-AUC"], marker="o",color=TEAL)
ax.scatter(["R2op"],[.23434],s=75,color=RED,zorder=3)
ax.set(ylabel="Recorded public PR-AUC",title="Selected public submission history (subject to selection bias)")
fig.tight_layout();fig.savefig(FIGURES / "04_public_history.png",bbox_inches="tight");plt.show()
pre = details["ranking"]
assert np.array_equal(np.argsort(pre,kind="stable"),np.argsort(original_p,kind="stable"))
print(f"Before final operating shift: {int((pre>=.5).sum())} rows above 0.5; after: {int((original_p>=.5).sum())} rows.")
print(f"Output threshold 0.5 corresponds to pre-shift score {expit(-2.188389):.6f}; it is fixed for every future batch.")
manifest=json.loads((PACKAGE / "manifest.json").read_text())
validation=json.loads((ROOT / "experiments" / "codex" / "package_validation.json").read_text())
display(pd.DataFrame(validation["isolated_notebook_cases"])[["case","rows","max_difference","passed","training_file_present"]])
assert validation["passed"]
out=pd.DataFrame({"id":te.id,"prediction":reproduced_p})
(ROOT / "tmp").mkdir(exist_ok=True)
out.to_csv(ROOT / "tmp" / "judge_reproduced_predictions.csv",index=False)
print("Freshly trained predictions saved to tmp/judge_reproduced_predictions.csv; upload package remains unchanged.")''')
    md('''A constant logit shift preserves ranking/PR-AUC mathematically; it changes score calibration. R2op applies +2.188389 so its 0.5 threshold flags 300 public-test rows. It does **not** recalculate a top-2.5% threshold on future batches: the threshold and reference logit stay fixed, preserving singleton and batch independence.

The handbook names PR-AUC, F1 and recall. The portal specifies fraud probabilities and PR-AUC but does not state the private decision threshold. We preserve the public winner's existing operating output and provide `predictions_before_operating_shift.csv` as an explicitly separate reference. The latter still includes public-tuned adjustments and is not guaranteed calibrated. We cannot assert improved private F1/recall from either file without the evaluator rule and test labels.

## 6. Limitations and judge Q&A

| Question | Evidence-bound answer |
|---|---|
| Why LR? | With 353 positives, stable regularisation and engineered relationships beat the tested higher-capacity alternatives; GALR supplies modest complementary evidence. |
| How were shifts chosen? | Distributional hypotheses followed by public-board probes. Their sizes are public-tuned and may overfit. High-value promotions have weaker data-only support. |
| Are the boxes proven injected/random-label rows? | No. Hour/support patterns support synthetic additions; exact membership and label mechanisms are unknown. |
| Is T2 exactly 3.5% fraudulent? | No. It is a working level indirectly bracketed from aggregate public probes. Test labels are unavailable. |
| Can the private population differ? | Yes. Unsupervised public-test centering and fixed public-tuned rules are dependencies on this population, even though runtime is row-local. |
| Did the residual scan prove a 0.24-0.25 ceiling? | No. Failure to find a useful residual feature does not bound attainable performance. |
| Are fresh seed gains independent? | No. They use the same training rows and labels under different folds. Conditional row bootstrap uncertainty also excludes retraining variability. |
| Do final scores mean a calibrated fraud chance? | No. Public shifts and the operating shift change calibration. They are ranking/decision scores in [0,1]. |
| What would deployment require? | Representative future labeled validation, a business review budget, calibrated probabilities, monitoring of drift/false alarms and a review of which shift rules remain justified. |

**Outcome of this branch:** no credible new predictive winner was established. R2op is retained, its training recipe is reproduced, its delivery is robust to empty and malformed inputs, and its explanation separates evidence from uncertainty.''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"}})
    target = ROOT / "analysis" / "Track2_Judge_Analysis.ipynb"
    target.parent.mkdir(exist_ok=True)
    nbf.write(notebook, target)
    print(target)


if __name__ == "__main__":
    build()
