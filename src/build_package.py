"""python src/build_package.py <name> : fit chosen model on full train, write packages/<name>/{model.pkl, prediction_notebook.ipynb} and submissions/sub_<name>.csv"""
import sys, os, joblib, subprocess, pandas as pd
sys.path.insert(0, "src"); from common import load; import models
name = sys.argv[1]
BUILDERS = {"v2_lr_hardened": lambda tr: models.fit_lr(tr, C=0.1),
            "v3_lr": lambda tr: models.fit_lr(tr, C=0.1, num=models.LR_NUM_V3),
            "v4_lr": lambda tr: models.fit_lr(tr, C=0.1, num=models.LR_NUM_V4)}
tr, te = load(); b = BUILDERS[name](tr)
os.makedirs(f"packages/{name}", exist_ok=True)
for f in os.listdir(f"packages/{name}"): os.remove(f"packages/{name}/{f}")
joblib.dump(b, f"packages/{name}/model.pkl")
p = models.predict_bundle(b, te)
pd.DataFrame({"id": te.id, "prediction": p}).to_csv(f"submissions/sub_{name}.csv", index=False)
subprocess.run([sys.executable, "src/make_notebook.py", f"packages/{name}"], check=True)
print("built", name, "pred mean", p.mean())
