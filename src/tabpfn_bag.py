"""TabPFN context bagging: each context = all fold-train frauds + a random 8,000 legit rows; predictions averaged over contexts.
Raw + engineered features (the better TabPFN variant). One repeat of evaluate.SPLITS; cached per fold."""
import sys, os, time, numpy as np, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("TABPFN_NO_BROWSER", "1")
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0, "src")
from evaluate import TR, Y, SPLITS
from tabpfn_cv import frame
from gpulock import gpu_lock
N_CTX, N_NEG = int(sys.argv[1]) if len(sys.argv) > 1 else 6, 8000

if __name__ == "__main__":
    from tabpfn import TabPFNClassifier
    cache = f"experiments/tabpfn_bag{N_CTX}_rep0_folds.npz"; done = dict(np.load(cache)) if os.path.exists(cache) else {}
    with gpu_lock("track2-tabpfn-bag") as got:
        if not got: print("GPU busy - retry later"); sys.exit(1)
        for k, (a, b) in enumerate(SPLITS[:5]):
            if f"f{k}" in done: continue
            t = time.time(); A, B = TR.iloc[a], TR.iloc[b]; XA, XB, cat_idx = frame(A, B, True); ya = A.fraud.values
            rng = np.random.default_rng(100 + k); pos = np.flatnonzero(ya == 1); neg = np.flatnonzero(ya == 0); acc = np.zeros(len(XB))
            for c in range(N_CTX):
                idx = np.r_[pos, rng.choice(neg, N_NEG, replace=False)]
                m = TabPFNClassifier(device="cuda", n_estimators=4, random_state=c, categorical_features_indices=cat_idx, ignore_pretraining_limits=True)
                m.fit(XA[idx], ya[idx])
                for s in range(0, len(XB), 2000): acc[s:s + 2000] += np.log(np.clip(m.predict_proba(XB[s:s + 2000])[:, 1], 1e-9, 1)) / N_CTX
            done[f"f{k}"] = np.exp(acc); np.savez(cache, **done)
            print(f"fold {k}: {time.time() - t:.0f}s AP {ap(Y[b], done[f'f{k}']):.4f}", flush=True)
    oof = np.full((5, len(Y)), np.nan); oof[0] = 0
    for k, (a, b) in enumerate(SPLITS[:5]): oof[0, b] = done[f"f{k}"]
    np.save(f"experiments/oof5_tabpfn_bag{N_CTX}.npy", oof)
    import arch_eval; arch_eval.evaluate(oof, f"tabpfn_bag{N_CTX}")
