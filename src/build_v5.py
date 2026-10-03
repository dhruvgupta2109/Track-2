"""Build v5 packages. v5_lr_knn: v4 + age<120 + context amount + kNN distance ratio. v5a_lr: without kNN.
Training rows get leave-one-out kNN distances (self excluded), exactly as in CV; inference uses the stored reference sets."""
import sys, os, joblib, subprocess, numpy as np, pandas as pd
sys.path.insert(0, "src")
from common import load
import models
from features_prod import make_features, add_v5_features, V5_EMB
from constants import fit_constants
from sklearn.neighbors import NearestNeighbors

def fit_v5(tr, knn=True):
    K = fit_constants(tr); A = make_features(tr, K); y = tr.fraud.values
    K["global_med_log_amt"] = float(A.log_amt.median())
    for g in ["transaction_channel", "country"]:
        K["med_log_amt_" + g] = A.groupby(g).log_amt.median().to_dict()
    K["sorted_log_amt_by_cat"] = {c: np.sort(s.dropna().values) for c, s in A.groupby("merchant_category").log_amt}
    num = list(models.LR_NUM_V4) + ["age_lt120", "amt_rel_transaction_channel", "amt_rel_country", "amt_pct_cat"]
    if knn:
        med = A[V5_EMB].median(); X = A[V5_EMB].fillna(med).values
        K["emb_median"] = med.to_dict(); K["emb_mean"] = X.mean(0); K["emb_scale"] = X.std(0)
        E = (X - K["emb_mean"]) / K["emb_scale"]
        K["nn_fraud"] = NearestNeighbors(n_neighbors=2).fit(E[y == 1]); K["nn_legit"] = NearestNeighbors(n_neighbors=2).fit(E[y == 0])
        num.append("d_ratio")
    A = add_v5_features(A, K, knn=False)
    if knn:   # leave-one-out distances for the training rows
        df_ = K["nn_fraud"].kneighbors(E)[0]; dl_ = K["nn_legit"].kneighbors(E)[0]
        d_f = np.where(y == 1, df_[:, 1], df_[:, 0]); d_l = np.where(y == 0, dl_[:, 1], dl_[:, 0])
        A["d_ratio"] = np.log1p(d_f) - np.log1p(d_l)
    m = models.lr_pipeline(0.1, num, models.LR_CAT).fit(A[models.LR_CAT + num], y)
    return {"model": m, "features": models.LR_CAT + num, "constants": K, "prior": float(y.mean()), "v5_knn": knn}

def predict_v5(b, df):
    X = add_v5_features(make_features(df, b["constants"]), b["constants"], knn=b["v5_knn"])
    return b["model"].predict_proba(X[b["features"]])[:, 1]

if __name__ == "__main__":
    tr, te = load()
    for name, knn in [("v5_lr_knn", True), ("v5a_lr", False)]:
        os.makedirs(f"packages/{name}", exist_ok=True)
        for f in os.listdir(f"packages/{name}"): os.remove(f"packages/{name}/{f}")
        b = fit_v5(tr, knn); joblib.dump(b, f"packages/{name}/model.pkl")
        p = predict_v5(b, te); pd.DataFrame({"id": te.id, "prediction": p}).to_csv(f"submissions/sub_{name}.csv", index=False)
        subprocess.run([sys.executable, "src/make_notebook.py", f"packages/{name}", "v5"], check=True)
        print(name, "built; mean pred", p.mean().round(5), "pkl MB", round(os.path.getsize(f"packages/{name}/model.pkl") / 1e6, 2))
