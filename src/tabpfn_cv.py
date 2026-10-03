"""TabPFN (pretrained tabular transformer) on the exact evaluate.SPLITS folds, compared with v5 and the R2 base blend on the same repeat.
Per-fold predictions are cached so an interrupted run resumes."""
import sys, os, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("TABPFN_NO_BROWSER", "1")
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, "src")
from evaluate import TR, Y, SPLITS
from features_prod import make_features
from constants import fit_constants
from gpulock import gpu_lock

CATS = ["merchant_category", "country", "transaction_channel"]
RAW = ["transaction_amount", "transaction_hour", "transactions_last_24h", "spend_last_24h", "account_age", "new_device", "transactions_last_1h"]
ENG = ["log_amt", "amt_rel_cat", "avg_spend", "amt_rel_avg", "night", "foreign", "burst", "young", "t1_share"]
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))


def frame(a, b, eng):
    K = fit_constants(a); A, B = make_features(a, K), make_features(b, K)
    cols = RAW + (ENG if eng else [])
    XA, XB = A[cols].astype(float).copy(), B[cols].astype(float).copy()
    for c in CATS:   # ordinal codes from the training fold; unseen/missing -> NaN
        lv = {v: i for i, v in enumerate(sorted(A[c].dropna().unique()))}
        XA[c] = A[c].map(lv).astype(float); XB[c] = B[c].map(lv).astype(float)
    return XA.values, XB.values, [len(cols) + i for i in range(len(CATS))]


def fit_predict(a, b, eng, n_est=8, seed=0):
    from tabpfn import TabPFNClassifier
    XA, XB, cat_idx = frame(a, b, eng)
    m = TabPFNClassifier(device="cuda", n_estimators=n_est, random_state=seed, categorical_features_indices=cat_idx, ignore_pretraining_limits=True)
    m.fit(XA, a.fraud.values)
    out = np.zeros(len(XB))
    for s in range(0, len(XB), 2000): out[s:s + 2000] = m.predict_proba(XB[s:s + 2000])[:, 1]
    return out


if __name__ == "__main__":
    eng = (sys.argv[1] == "eng") if len(sys.argv) > 1 else False; rep = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tag = f"tabpfn_{'eng' if eng else 'raw'}_rep{rep}"; cache = f"experiments/{tag}_folds.npz"
    done = dict(np.load(cache)) if os.path.exists(cache) else {}
    with gpu_lock("track2-tabpfn") as got:
        if not got: print("GPU busy (lock held) - not running on CPU; retry later"); sys.exit(1)
        for k, (a, b) in enumerate(SPLITS[rep * 5:(rep + 1) * 5]):
            if f"f{k}" in done: continue
            t = time.time(); done[f"f{k}"] = fit_predict(TR.iloc[a], TR.iloc[b], eng); np.savez(cache, **done)
            print(f"{tag} fold {k}: {time.time() - t:.0f}s, fold AP {ap(Y[b], done[f'f{k}']):.4f}", flush=True)
    oof = np.zeros(len(Y))
    for k, (a, b) in enumerate(SPLITS[rep * 5:(rep + 1) * 5]): oof[b] = done[f"f{k}"]
    np.save(f"experiments/oof_{tag}.npy", oof)
    O5 = np.load("experiments/oof5_v5_prod.npy")[rep]; PO = np.load("experiments/oof5_bl_poly.npy")[rep]; GA = np.load("experiments/oof5_galr_C0.05_int.npy")[rep]
    r2 = 0.8 * (0.7 * lg(O5) + 0.3 * lg(PO)) + 0.2 * lg(GA)
    print(f"REPEAT {rep}: TabPFN {'raw+eng' if eng else 'raw'} AP {ap(Y, oof):.4f} | v5 {ap(Y, O5):.4f} | R2 base blend {ap(Y, r2):.4f}")
    zt = lg(oof); zt = (zt - zt.mean()) / zt.std() * r2.std()
    for w in (0.1, 0.2, 0.3, 0.5, 0.7):
        print(f"   R2 base + TabPFN w{w}: {ap(Y, (1 - w) * r2 + w * zt):.4f} ({ap(Y, (1 - w) * r2 + w * zt) - ap(Y, r2):+.4f})")
