"""R3op package = R2op + degree-3 interaction LR (C=0.001) at 10% of the logit (fresh-seed CV +0.0002/+0.0051/+0.0016/+0.0028).
Two notebook passes in the clean env: (1) T2/operating-point constants neutral -> pre-T2 logits; (2) constants fitted, final run."""
import sys, os, re, json, shutil, subprocess, hashlib, joblib, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, "src")
from blendlab import poly_consts, poly_frame, poly_pipe
from templates import template_masks
SRC = "../Track2_R2op_package"; DST = "../Track2_R3op_package"; CLEAN_PY = os.path.abspath(".venv_clean/Scripts/python.exe")
LEVEL, ALPHA, W3 = 0.035, 0.3, 0.1
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv")

bundle = joblib.load(f"{SRC}/model.pkl")
PC = poly_consts(tr)
bundle["poly3_model"] = poly_pipe(0.001, 3).fit(poly_frame(tr, PC), tr.fraud.values)
bundle["poly3_constants"] = {"cat_med_log": PC["cat_med_log"], "risk_maps": PC["risk"], "risk_prior": PC["prior"],
                             "core": ["log_amt", "log_spend", "log_age", "transactions_last_24h", "transactions_last_1h", "new_device", "amt_vs_cat",
                                      "foreign", "young", "burst", "night"], "cat": ["merchant_category", "country", "transaction_channel"]}
bundle["blend_w_poly3"] = W3
if os.path.exists(DST): shutil.rmtree(DST)
os.makedirs(DST); joblib.dump(bundle, f"{DST}/model.pkl"); shutil.copy(f"{SRC}/requirements.txt", DST)

nb = json.load(open(f"{SRC}/prediction_notebook.ipynb"))
cell = [x for x in nb["cells"] if "W_GALR = bundle" in "".join(x["source"])][0]; src = "".join(cell["source"])
old = "preds = 1.0 / (1.0 + np.exp(-((1 - W_GALR) * z_q30 + W_GALR * to_logit(p_galr))))   # ensemble probability"
assert old in src
new = "\n".join([
    "z_r2 = (1 - W_GALR) * z_q30 + W_GALR * to_logit(p_galr)",
    "p_poly3 = bundle[\"poly3_model\"].predict_proba(poly_features(X_all, bundle[\"poly3_constants\"]))[:, 1]   # model 4: degree-3 interaction LR",
    "W_POLY3 = bundle[\"blend_w_poly3\"]                                                   # 0.1 (4/4 fresh CV seeds positive)",
    "preds = 1.0 / (1.0 + np.exp(-((1 - W_POLY3) * z_r2 + W_POLY3 * to_logit(p_poly3))))   # ensemble probability"])
cell["source"] = src.replace(old, new).splitlines(keepends=True)
md = nb["cells"][0]
md["source"] = "".join(md["source"]) + ("\n\n**Model 4 (R3)**: degree-3 interaction logistic regression (C=0.001) on the same 14 risk features, "
                                        "10% of the logit (fresh-seed CV +0.0002 / +0.0051 / +0.0016 / +0.0028).")
tcell = [i for i, x in enumerate(nb["cells"]) if "T2_LEVEL" in "".join(x["source"])][0]


def write(ref_logit, op):
    t = "".join(nb["cells"][tcell]["source"])
    t = re.sub(r"T2_LEVEL, T2_ALPHA, T2_REF_LOGIT = .*", f"T2_LEVEL, T2_ALPHA, T2_REF_LOGIT = {LEVEL}, {ALPHA}, {ref_logit:.6f}", t)
    t = re.sub(r"OPERATING_POINT_SHIFT = .*", f"OPERATING_POINT_SHIFT = {op:.6f}", t)
    nb["cells"][tcell]["source"] = t.splitlines(keepends=True)
    for x in nb["cells"]:
        if x["cell_type"] == "code": x["outputs"] = []; x["execution_count"] = None
    json.dump(nb, open(f"{DST}/prediction_notebook.ipynb", "w"), indent=1)


def run():
    d = os.path.abspath("../_r3run"); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for f in ("model.pkl", "prediction_notebook.ipynb", "requirements.txt"): shutil.copy(f"{DST}/{f}", d)
    shutil.copy("test.csv", d)
    r = subprocess.run([CLEAN_PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--output", "executed.ipynb", "prediction_notebook.ipynb"],
                       cwd=d, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    out = pd.read_csv(f"{d}/predictions.csv"); shutil.rmtree(d); return out


lg = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12)))
T2 = template_masks(te)["T2"]
write(0.0, 0.0); z1 = lg(run().prediction.values)
pre = z1.copy(); pre[T2] = (z1[T2] - lg(np.array([LEVEL]))[0]) / ALPHA
REF = float(pre[T2].mean()); z = pre.copy(); z[T2] = lg(np.array([LEVEL]))[0] + ALPHA * (pre[T2] - REF)
OP = float(-lg(np.array([np.quantile(1 / (1 + np.exp(-z)), 1 - 0.025)]))[0])
write(REF, OP); out = run()
out.to_csv(f"{DST}/predictions_R3op.csv", index=False); out.to_csv("submissions/tmpl/R3op_package_output.csv", index=False)
q = pd.read_csv("submissions/tmpl/R2op_package_output.csv").prediction.values
print(f"REF {REF:.4f} OP {OP:.4f}; flagged >= 0.5: {(out.prediction >= 0.5).sum()}; spearman vs R2op {spearmanr(out.prediction, q)[0]:.4f}; "
      f"sha {hashlib.sha256(open('submissions/tmpl/R3op_package_output.csv', 'rb').read()).hexdigest()[:10]}; cols {list(out.columns)} rows {len(out)}")
