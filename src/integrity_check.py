"""Automated submission-integrity test: python src/integrity_check.py <package_dir> <submission_csv>"""
import sys, os, shutil, tempfile, subprocess, joblib, numpy as np, pandas as pd
sys.path.insert(0, "src")
pkg, sub = sys.argv[1], sys.argv[2]
te = pd.read_csv("test.csv"); ss = pd.read_csv("sample_submission.csv"); tr = pd.read_csv("train.csv"); s = pd.read_csv(sub)
R = []
def chk(name, cond): R.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name)
chk("1 columns exactly id,prediction", list(s.columns) == ["id", "prediction"])
chk("2 ids identical & same order as sample_submission", (s.id.values == ss.id.values).all())
chk("3 ids identical & same order as test.csv", (s.id.values == te.id.values).all())
chk("4 row count == test rows (12000)", len(s) == len(te) == 12000)
chk("5 no duplicate / missing ids", s.id.is_unique and s.id.notna().all())
chk("6 prediction numeric, finite, in [0,1]", pd.api.types.is_float_dtype(s.prediction) and np.isfinite(s.prediction).all() and s.prediction.between(0, 1).all())
chk("7 train target: 1 = fraud (values {0,1}, minority=1)", set(tr.fraud.unique()) == {0, 1} and tr.fraud.mean() < 0.05)
b = joblib.load(os.path.join(pkg, "model.pkl")); m = b["model"]
print("   model.classes_ =", m.classes_)
chk("8 classes_ == [0,1] so predict_proba[:,1] is P(fraud)", list(m.classes_) == [0, 1])
exec(open("src/features_prod.py").read())
X = make_features(te, b["constants"])[b["features"]]
p = m.predict_proba(X)[:, list(m.classes_).index(1)]
chk("9 csv == fresh in-process predictions (by id)", np.abs(p - s.set_index("id").loc[te.id, "prediction"].values).max() < 1e-12)
mean_fraud = s.prediction.mean(); prior = tr.fraud.mean()
chk("10 probability direction: mean pred ~ prior (not ~1-prior)", 0.3 * prior < mean_fraud < 3 * prior)
# direction on train: predictions should be higher for fraud rows
pt = m.predict_proba(make_features(tr, b["constants"])[b["features"]])[:, 1]
chk("11 train in-sample: mean P(fraud|fraud) > mean P(fraud|legit)", pt[tr.fraud == 1].mean() > 3 * pt[tr.fraud == 0].mean())
chk("12 feature order stored in artifact equals pipeline input order", list(m.feature_names_in_) == list(b["features"]))
# reordered / extra columns / missing column robustness
rev = te[te.columns[::-1]].assign(junk=1)
p2 = m.predict_proba(make_features(rev, b["constants"])[b["features"]])[:, 1]
chk("13 column reorder + extra col -> identical preds", np.abs(p2 - p).max() < 1e-12)
perm = np.random.default_rng(3).permutation(len(te)); p3 = m.predict_proba(make_features(te.iloc[perm].drop(columns="id"), b["constants"])[b["features"]])[:, 1]
chk("14 shuffled id-less input -> same preds row-aligned", np.abs(p3 - p[perm]).max() < 1e-12)
cats = {c: sorted(m.named_steps["columntransformer"].named_transformers_["c"].named_steps["onehotencoder"].categories_[i]) for i, c in enumerate(["merchant_category", "country", "transaction_channel", "hc"])}
chk("15 encoder categories == train categories (+NA)", all(set(cats[c]) - {"NA"} == set(tr[c].dropna().unique()) for c in ["merchant_category", "country", "transaction_channel"]))
print("ALL PASS" if all(r[1] for r in R) else "FAILURES PRESENT")
