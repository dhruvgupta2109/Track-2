"""ChatGPT-proposed generator formula: evaluate as-is and as a CV-refitted basis (honest)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from sklearn.metrics import average_precision_score as ap
from sklearn.linear_model import LogisticRegression
MER = {"grocery": 0.0, "retail": 0.0, "transport": 0.0, "gaming": 0.2, "entertainment": 0.3, "utilities": 0.4, "food_delivery": 0.5, "travel": 0.6, "electronics": 0.9, "luxury": 1.1, "cash_transfer": 1.2}
CTY = {"MY": -0.3, "TH": -0.1, "SG": 0.0, "AU": 0.0, "JP": 0.2, "US": 0.3, "GB": 0.3, "VN": 0.4, "PH": 0.6, "ID": 0.7}
CHN = {"card_present": 0.0, "mobile_app": 0.4, "bank_transfer": 0.5, "ecommerce": 0.6}

def basis(df):
    """Generator terms; missing inputs left NaN (handled by caller)."""
    a, n1, n24 = df.account_age, df.transactions_last_1h, df.transactions_last_24h
    B = pd.DataFrame(index=df.index)
    B["nd"] = df.new_device
    B["amt500"] = (df.transaction_amount > 500).astype(float)
    B["night"] = df.transaction_hour.isin([23, 0, 1, 2, 3, 4]).astype(float)
    B["age90"] = (a < 90).astype(float).where(a.notna()); B["age90_180"] = ((a >= 90) & (a < 180)).astype(float).where(a.notna())
    B["age180_365"] = ((a >= 180) & (a < 365)).astype(float).where(a.notna())
    B["n1x"] = np.clip(n1 - 2, 0, 3); B["n24x"] = np.maximum(n24 - 7, 0)
    for k in MER: B["m_" + k] = (df.merchant_category == k).astype(float).where(df.merchant_category.notna())
    for k in CTY: B["c_" + k] = (df.country == k).astype(float).where(df.country.notna())
    for k in CHN: B["h_" + k] = (df.transaction_channel == k).astype(float).where(df.transaction_channel.notna())
    return B

COEF = {"nd": 1.5, "amt500": 0.9, "night": 0.6, "age90": 1.0, "age90_180": 0.8, "age180_365": 0.5, "n1x": 0.4, "n24x": 0.05,
        **{"m_" + k: v for k, v in MER.items()}, **{"c_" + k: v for k, v in CTY.items()}, **{"h_" + k: v for k, v in CHN.items()}}

def formula_logit(df, train_ref):
    """Missing inputs -> expected value of the term under the training distribution (labels drawn before masking)."""
    B = basis(df); M = basis(train_ref).mean()
    B = B.fillna(M)
    return -6.0 + sum(COEF[c] * B[c] for c in COEF)

if __name__ == "__main__":
    tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv"); y = tr.fraud.values
    z = formula_logit(tr, tr); print("formula as-is on train: AP %.4f (in-sample for its author)" % ap(y, z))
    # calibration slope check
    lr = LogisticRegression(C=1e6).fit(z.values[:, None], y); print("calibration: intercept %.3f slope %.3f" % (lr.intercept_[0], lr.coef_[0, 0]))
    o5 = np.load("experiments/oof5_v5_prod.npy"); print("v5 OOF AP (5 reps):", round(np.mean([ap(y, o5[r]) for r in range(5)]), 4))
    # honest: refit the formula's basis by LR inside the same CV folds as v5
    from evaluate import SPLITS
    oof = np.zeros((5, len(y)))
    for i, (a_, b_) in enumerate(SPLITS):
        A, Bv = basis(tr.iloc[a_]), basis(tr.iloc[b_]); m = A.mean(); A, Bv = A.fillna(m), Bv.fillna(m)
        oof[i // 5, b_] = LogisticRegression(C=1.0, max_iter=3000).fit(A, y[a_]).predict_proba(Bv)[:, 1]
    print("formula basis refitted in-fold: AP %.4f" % np.mean([ap(y, oof[r]) for r in range(5)]))
    np.save("experiments/oof5_genbasis.npy", oof)
    # test: what does the formula imply for bursts / nd-old?
    zt = formula_logit(te, tr).values; pt = 1 / (1 + np.exp(-zt))
    v5 = pd.read_csv("submissions/sub_v5_lr_knn.csv").prediction.values; lg = lambda p: np.log(p / (1 - p))
    burst = (te.transactions_last_24h >= 16).values; ndold = ((te.new_device == 1) & (te.account_age > 365)).values & ~burst
    d = zt - lg(v5)
    print("test mean logit(formula) - logit(v5): all %.2f | bursts %.2f | nd&old %.2f | rest %.2f" % (d.mean(), d[burst].mean(), d[ndold].mean(), d[~burst & ~ndold].mean()))
    print("formula test mean p %.4f" % pt.mean())
    pd.DataFrame({"id": te.id, "prediction": pt}).to_csv("submissions/shift/G1_generator_formula.csv", index=False)
