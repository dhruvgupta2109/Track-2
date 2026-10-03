"""R2op package = R1op + GALR (generator-aware basis LR, unweighted L2, C=0.05) blended at 20% of the logit before the group shifts.
Two passes: (1) notebook with T2/operating-point constants neutral -> pre-T2 logits from the clean env; (2) constants fitted, final run."""
import sys, os, json, shutil, subprocess, inspect, joblib, numpy as np, pandas as pd
sys.path.insert(0, "src")
import galr
from templates import template_masks
SRC = "../Track2_R1op_package"; DST = "../Track2_R2op_package"
CLEAN_PY = os.path.abspath(".venv_clean/Scripts/python.exe")
C_GALR, W_GALR, LEVEL, ALPHA = 0.05, 0.2, 0.035, 0.3
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv")

# model bundle: friend's Q30 bundle + GALR
bundle = joblib.load(f"{SRC}/model.pkl")
A = galr.basis(tr); num = [c for c in A.columns if c not in galr.CAT]
bundle["galr_model"] = galr.pipe(num, C_GALR).fit(A, tr.fraud.values); bundle["galr_columns"] = list(A.columns); bundle["blend_w_galr"] = W_GALR
if os.path.exists(DST): shutil.rmtree(DST)
os.makedirs(DST); joblib.dump(bundle, f"{DST}/model.pkl"); shutil.copy(f"{SRC}/requirements.txt", DST)

nb = json.load(open(f"{SRC}/prediction_notebook.ipynb"))
basis_src = (inspect.getsource(galr.basis).replace("AMT_T", "GALR_AMT_T").replace("AGE_T", "GALR_AGE_T").replace("NIGHT", "GALR_NIGHT").replace("CAT[:3]", "GALR_CAT[:3]"))
GALR_CELL = ("# Model 3: generator-aware basis logistic ranker (unweighted L2 LR on step indicators at plausible synthetic thresholds).\n"
             "GALR_AMT_T = [100, 200, 300, 500, 750, 1000, 1500, 2500]; GALR_AGE_T = [120, 270, 365]; GALR_NIGHT = [23, 0, 1, 2, 3, 4]\n"
             "GALR_CAT = [\"merchant_category\", \"country\", \"transaction_channel\", \"hour_c\"]\n" + basis_src)
nb["cells"].insert(4, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": GALR_CELL.splitlines(keepends=True)})
c = nb["cells"][5]; s = "".join(c["source"])
old = "preds = 1.0 / (1.0 + np.exp(-((1 - BLEND_W_POLY) * to_logit(p_v5) + BLEND_W_POLY * to_logit(p_poly))))   # ensemble probability"
assert old in s
new = ("z_q30 = (1 - BLEND_W_POLY) * to_logit(p_v5) + BLEND_W_POLY * to_logit(p_poly)\n"
       "p_galr = bundle[\"galr_model\"].predict_proba(basis(X_all)[bundle[\"galr_columns\"]])[:, 1]   # model 3: generator-aware basis LR\n"
       "W_GALR = bundle[\"blend_w_galr\"]                                                     # 0.2 (nested-CV selected range 0.1-0.3)\n"
       "preds = 1.0 / (1.0 + np.exp(-((1 - W_GALR) * z_q30 + W_GALR * to_logit(p_galr))))   # ensemble probability")
c["source"] = s.replace(old, new).splitlines(keepends=True)
md = nb["cells"][0]; md["source"] = "".join(md["source"]) + ("\n\n**Model 3 (R2)**: generator-aware basis logistic ranker (step indicators at amount 100-2500, account age 120/270/365, "
      "hour one-hot + night, 1h activity, clipped/excess 24h count, new_device x night / x ecommerce), unweighted L2 LR (C=0.05), blended at 20% of the logit. "
      "Nested CV: +0.0017 / +0.0033 PR-AUC over the two-model blend.")
tcell = [i for i, x in enumerate(nb["cells"]) if "T2_LEVEL" in "".join(x["source"])][0]


def write(ref_logit, op):
    t = "".join(nb["cells"][tcell]["source"])
    import re
    t = re.sub(r"T2_LEVEL, T2_ALPHA, T2_REF_LOGIT = .*", f"T2_LEVEL, T2_ALPHA, T2_REF_LOGIT = {LEVEL}, {ALPHA}, {ref_logit:.6f}", t)
    t = re.sub(r"OPERATING_POINT_SHIFT = .*", f"OPERATING_POINT_SHIFT = {op:.6f}", t)
    nb["cells"][tcell]["source"] = t.splitlines(keepends=True)
    for x in nb["cells"]:
        if x["cell_type"] == "code": x["outputs"] = []; x["execution_count"] = None
    json.dump(nb, open(f"{DST}/prediction_notebook.ipynb", "w"), indent=1)


def run():
    d = os.path.abspath("../_r2run"); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for f in ("model.pkl", "prediction_notebook.ipynb", "requirements.txt"): shutil.copy(f"{DST}/{f}", d)
    shutil.copy("test.csv", d)
    r = subprocess.run([CLEAN_PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--output", "executed.ipynb", "prediction_notebook.ipynb"], cwd=d, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    out = pd.read_csv(f"{d}/predictions.csv"); shutil.rmtree(d); return out


lg = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12)))
T2 = template_masks(te)["T2"]
# pass 1: neutral T2 (alpha=1, level = ref -> identity) is not expressible exactly; use op=0 and read pre-T2 logits from rows outside T2,
# and get T2 rows' pre-T2 logits by inverting the T2 map with the pass-1 constants.
write(0.0, 0.0); o1 = run(); z1 = lg(o1.prediction.values)
pre = z1.copy(); pre[T2] = (z1[T2] - lg(np.array([LEVEL]))[0]) / ALPHA + 0.0
REF = float(pre[T2].mean())
z = pre.copy(); z[T2] = lg(np.array([LEVEL]))[0] + ALPHA * (pre[T2] - REF)
OP = float(-lg(np.array([np.quantile(1 / (1 + np.exp(-z)), 1 - 0.025)]))[0])
write(REF, OP); out = run()
out.to_csv(f"{DST}/predictions_R2op.csv", index=False); out.to_csv("submissions/tmpl/R2op_package_output.csv", index=False)
q = pd.read_csv("submissions/tmpl/R1op_package_output.csv").prediction.values
from scipy.stats import spearmanr
import hashlib
print(f"REF {REF:.4f} OP {OP:.4f}; flagged >= 0.5: {(out.prediction >= 0.5).sum()}; spearman vs R1op {spearmanr(out.prediction, q)[0]:.4f}; "
      f"sha {hashlib.sha256(open('submissions/tmpl/R2op_package_output.csv', 'rb').read()).hexdigest()[:10]}; cols {list(out.columns)} rows {len(out)}")
