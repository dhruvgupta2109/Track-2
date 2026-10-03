"""Platform simulation. Copies ONLY the upload files into a fresh temp dir and executes the notebook in a fresh kernel:
 (a) platform mode: shuffled test without id via env vars; (b) template default mode: test.csv with id next to notebook;
 (c) edge-case file: odd values / unknown categories / missing columns. Verifies outputs."""
import sys, os, shutil, tempfile, subprocess, numpy as np, pandas as pd
pkg, ref_csv = sys.argv[1], sys.argv[2]; py = sys.argv[3] if len(sys.argv) > 3 else sys.executable
uploads = [f for f in os.listdir(pkg) if not f.startswith("_")]
print("uploads:", uploads)
ok = True
def run(d, env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("DATATHON")}; env.update(env_extra); env["MPLBACKEND"] = "Agg"
    r = subprocess.run([py, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--output", "executed.ipynb",
                        "prediction_notebook.ipynb"], cwd=d, env=env, capture_output=True, text=True)
    if r.returncode: print("NBCONVERT FAILED", r.stderr[-1500:])
    return r.returncode == 0
def fresh():
    d = tempfile.mkdtemp(prefix="t2sbx_")
    for f in uploads: shutil.copy(os.path.join(pkg, f), d)
    return d
te = pd.read_csv("test.csv"); ref = pd.read_csv(ref_csv).set_index("id")["prediction"]
# (a)
d = fresh(); perm = np.random.default_rng(7).permutation(len(te)); shuf = te.iloc[perm].reset_index(drop=True)
shuf.drop(columns="id").to_csv(os.path.join(d, "hidden.csv"), index=False)
if run(d, {"DATATHON_INPUT_PATH": os.path.join(d, "hidden.csv"), "DATATHON_OUTPUT_PATH": os.path.join(d, "out.csv")}):
    o = pd.read_csv(os.path.join(d, "out.csv")); diff = np.abs(o.prediction.values - ref.loc[shuf.id].values).max()
    good = list(o.columns) == ["prediction"] and len(o) == len(te) and o.prediction.between(0, 1).all() and o.prediction.notna().all() and diff < 1e-9
    print("(a) platform mode:", list(o.columns), len(o), "maxdiff", diff, "OK" if good else "BAD"); ok &= good
else: ok = False
shutil.rmtree(d)
# (b)
d = fresh(); shutil.copy("test.csv", d)
if run(d, {}):
    o = pd.read_csv(os.path.join(d, "predictions.csv")); s = pd.read_csv("sample_submission.csv")
    good = list(o.columns) == ["id", "prediction"] and (o.id.values == s.id.values).all() and np.abs(o.prediction.values - ref.loc[o.id].values).max() < 1e-9
    print("(b) template default mode:", list(o.columns), len(o), "ids match sample:", (o.id.values == s.id.values).all(), "OK" if good else "BAD"); ok &= good
else: ok = False
shutil.rmtree(d)
# (c)
d = fresh()
edge = pd.DataFrame({"transaction_amount": [0, -5, 1e7, np.nan, 50, "abc"], "transaction_hour": [24, -1, np.nan, 3, 12, 7.6],
 "merchant_category": ["crypto", None, "luxury", "grocery", " Grocery ", ""], "country": ["XX", "SG", None, "SG", "sg", "SG"],
 "transaction_channel": ["atm", "ecommerce", "mobile_app", None, "ecommerce", "card_present"], "transactions_last_24h": [0, 0, 1000, np.nan, 3, 2],
 "spend_last_24h": [0, 10, 1e8, np.nan, -1, 5], "account_age": [0, -3, 1e5, np.nan, 10, 100], "new_device": [2, np.nan, 1, 0, 1, "yes"],
 "extra_col": 1})   # transactions_last_1h column missing entirely
edge.to_csv(os.path.join(d, "edge.csv"), index=False)
if run(d, {"DATATHON_INPUT_PATH": os.path.join(d, "edge.csv"), "DATATHON_OUTPUT_PATH": os.path.join(d, "out.csv")}):
    o = pd.read_csv(os.path.join(d, "out.csv")); good = len(o) == len(edge) and o.prediction.between(0, 1).all()
    print("(c) edge cases:", o.prediction.round(4).tolist(), "OK" if good else "BAD"); ok &= good
else: ok = False
shutil.rmtree(d)
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
