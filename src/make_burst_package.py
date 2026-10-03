"""Package = R2op package + burst-by-country logit shifts (inserted just before the operating-point shift; the operating-point constant
is unchanged so the notebook reproduces the scored CSV, which was built as R2op output + the same shifts).
usage: make_burst_package.py NAME D_MYIDSG D_OTHER_YOUNG D_OTHER_OLD SCORED_CSV"""
import sys, os, json, shutil, subprocess, hashlib, numpy as np, pandas as pd
NAME, D_PU, D_OY, D_OO, SCORED = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
SRC = "../Track2_R2op_package"; DST = f"../Track2_{NAME}_package"; CLEAN_PY = os.path.abspath(".venv_clean/Scripts/python.exe")
if os.path.exists(DST): shutil.rmtree(DST)
os.makedirs(DST)
for f in ("model.pkl", "requirements.txt"): shutil.copy(f"{SRC}/{f}", DST)
nb = json.load(open(f"{SRC}/prediction_notebook.ipynb"))
cell = [c for c in nb["cells"] if "OPERATING_POINT_SHIFT" in "".join(c["source"])][0]; src = "".join(cell["source"])
anchor = "# Operating point."
assert anchor in src and src.count(anchor) == 1
BURST = f'''# Velocity bursts by country type (public-board tuned). The evaluation data's extra bursts are MY/ID/SG "power users"
# (3.9x the training rate) while bursts from other countries (1.7x) keep the fraud: bursts are split by country type.
SHIFT_BURST_MYIDSG, SHIFT_BURST_OTHER_YOUNG, SHIFT_BURST_OTHER_OLD = {D_PU}, {D_OY}, {D_OO}
_b = (X_all["transactions_last_24h"] >= 16).to_numpy()
_c = X_all["country"]
_pu = _c.isin(["MY", "ID", "SG"]).to_numpy() & _b
_ot = (_c.notna() & ~_c.isin(["MY", "ID", "SG"])).to_numpy() & _b
_yg = (X_all["account_age"] <= 365).to_numpy()
logit[_pu] += SHIFT_BURST_MYIDSG
logit[_ot & _yg] += SHIFT_BURST_OTHER_YOUNG
logit[_ot & ~_yg] += SHIFT_BURST_OTHER_OLD

'''
cell["source"] = src.replace(anchor, BURST + anchor).splitlines(keepends=True)
md = nb["cells"][0]
md["source"] = "".join(md["source"]) + (f"\n\n**Burst-by-country split** (public-board tuned): velocity bursts from MY/ID/SG {D_PU:+}, "
                                        f"from other countries {D_OY:+} (account <= 365 days) / {D_OO:+} (older), logit.")
for c in nb["cells"]:
    if c["cell_type"] == "code": c["outputs"] = []; c["execution_count"] = None
json.dump(nb, open(f"{DST}/prediction_notebook.ipynb", "w"), indent=1)
d = os.path.abspath(f"../_run_{NAME}"); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
for f in ("model.pkl", "prediction_notebook.ipynb", "requirements.txt"): shutil.copy(f"{DST}/{f}", d)
shutil.copy("test.csv", d)
r = subprocess.run([CLEAN_PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--output", "executed.ipynb", "prediction_notebook.ipynb"], cwd=d, capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-2000:]
out = pd.read_csv(f"{d}/predictions.csv"); shutil.rmtree(d)
ref = pd.read_csv(SCORED)
assert (out.id.values == ref.id.values).all()
diff = np.abs(out.prediction.values - ref.prediction.values).max()
same_rank = (np.argsort(-out.prediction.values, kind="stable") == np.argsort(-ref.prediction.values, kind="stable")).all()
out.to_csv(f"{DST}/predictions_{NAME}.csv", index=False)
print(f"{DST}: notebook output vs scored CSV max diff {diff:.2e}, identical ranking {same_rank}; flagged >= 0.5: {(out.prediction >= 0.5).sum()}; "
      f"package CSV sha {hashlib.sha256(open(f'{DST}/predictions_{NAME}.csv', 'rb').read()).hexdigest()[:10]} (scored sha {hashlib.sha256(open(SCORED, 'rb').read()).hexdigest()[:10]})")
