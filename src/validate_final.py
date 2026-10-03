"""Portal-rule validation: platform mode (no id, shuffled), default mode, edge cases, determinism. Single `prediction` column only."""
import os, sys, shutil, subprocess, tempfile, numpy as np, pandas as pd
PKG = sys.argv[1]; PY = os.path.abspath(".venv_clean/Scripts/python.exe")
UP = ["prediction_notebook.ipynb", "model.pkl"]
ref = pd.read_csv(f"{PKG}/reference_C2_public_0.24174.csv"); te = pd.read_csv("test.csv"); assert (ref.id == te.id).all()
def run(files, env):
    d = tempfile.mkdtemp(prefix="final_"); [shutil.copy(f"{PKG}/{f}", d) for f in UP]
    for name, df in files.items(): df.to_csv(os.path.join(d, name), index=False)
    e = {k: v for k, v in os.environ.items() if not k.startswith("DATATHON")}; e.update({k: v.replace("{d}", d) for k, v in env.items()}); e["MPLBACKEND"] = "Agg"
    r = subprocess.run([PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--output", "x.ipynb", "prediction_notebook.ipynb"], cwd=d, env=e, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-1500:]); sys.exit("RUN FAILED")
    out = pd.read_csv(os.path.join(d, e.get("DATATHON_OUTPUT_PATH", os.path.join(d, "predictions.csv")))); shutil.rmtree(d); return out
ok = True
perm = np.random.default_rng(7).permutation(len(te)); sh = te.iloc[perm].drop(columns="id").reset_index(drop=True)
env = {"DATATHON_INPUT_PATH": "{d}/hidden.csv", "DATATHON_OUTPUT_PATH": "{d}/out/preds.csv"}
os.makedirs  # nested output dir tested below only if notebook creates it
env = {"DATATHON_INPUT_PATH": "{d}/hidden.csv", "DATATHON_OUTPUT_PATH": "{d}/preds.csv"}
a1 = run({"hidden.csv": sh}, env); a2 = run({"hidden.csv": sh}, env)
c = list(a1.columns) == ["prediction"] and len(a1) == len(te) and a1.prediction.between(0, 1).all() and a1.prediction.notna().all()
diff = np.abs(a1.prediction.values - ref.prediction.values[perm]).max(); det = np.array_equal(a1.prediction.values, a2.prediction.values)
print(f"(a) platform (no id, shuffled): columns {list(a1.columns)}, rows {len(a1)}, max diff vs scored C2 {diff:.1e}, deterministic re-run {det}"); ok &= c and diff < 1e-9 and det
b = run({"test.csv": te}, {})
c = list(b.columns) == ["prediction"] and len(b) == len(te) and np.abs(b.prediction.values - ref.prediction.values).max() < 1e-9
print(f"(b) default (test.csv with id, no env vars): columns {list(b.columns)}, rows {len(b)}, matches scored C2 in order {c}"); ok &= c
edge = pd.DataFrame({"transaction_amount": [0, -5, 1e7, np.nan, 50, "abc"], "transaction_hour": [24, -1, np.nan, 3, 12, 7.6],
 "merchant_category": ["crypto", None, "luxury", "grocery", " Grocery ", ""], "country": ["XX", "SG", None, "SG", "sg", "my"],
 "transaction_channel": ["atm", "ecommerce", "mobile_app", None, "ecommerce", "card_present"], "transactions_last_24h": [0, 0, 1000, np.nan, 3, 20],
 "spend_last_24h": [0, 10, 1e8, np.nan, -1, 5], "account_age": [0, -3, 1e5, np.nan, 10, 100], "new_device": [2, np.nan, 1, 0, 1, "yes"], "extra_col": 1})
e = run({"edge.csv": edge}, {"DATATHON_INPUT_PATH": "{d}/edge.csv", "DATATHON_OUTPUT_PATH": "{d}/o.csv"})
c = list(e.columns) == ["prediction"] and len(e) == len(edge) and e.prediction.between(0, 1).all()
print(f"(c) edge cases (missing column, junk values, unknown categories): {e.prediction.round(4).tolist()} {'OK' if c else 'BAD'}"); ok &= c
print("ALL PORTAL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
