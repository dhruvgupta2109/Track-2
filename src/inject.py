"""Hour fingerprint: injected test rows never use hours 5,6,7,21,22. Estimate the injected fraction of each segment."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from features_prod import make_features; from constants import fit_constants
CLEAN = [5, 6, 7, 21, 22]
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv"); K = fit_constants(tr); Dtr, Dte = make_features(tr, K), make_features(te, K)
def segs(D):
    a = D.account_age; nd = D.new_device == 1; b = D.burst == 1; t1 = D.transactions_last_1h
    return {"ALL": np.ones(len(D), bool), "burst": b, "burst_young": b & (a <= 365), "burst_old": b & ~(a <= 365),
            "nd_old(non-burst)": nd & (a > 365) & ~b, "nd_young<=90": nd & (a <= 90) & ~b, "nd_90-365": nd & (a > 90) & (a <= 365) & ~b,
            "t1h>=3 non-burst": (t1 >= 3) & ~b, "t24 10-15": (D.transactions_last_24h >= 10) & ~b,
            "amt>500": D.transaction_amount > 500, "amt>2000": D.transaction_amount > 2000, "big&age>1500": (D.amt_rel_cat > 1) & (a > 1500),
            "age>1500": a > 1500, "age<=90": a <= 90, "foreign": D.foreign == 1, "SG": D.foreign == 0, "no_nd&age>365&SG&amt<500": (~nd) & (a > 365) & (D.foreign == 0) & (D.transaction_amount < 500),
            **{f"cat={c}": D.merchant_category == c for c in ["cash_transfer", "electronics", "luxury", "travel", "grocery", "food_delivery", "retail", "transport", "gaming", "entertainment", "utilities"]},
            **{f"chan={c}": D.transaction_channel == c for c in ["card_present", "ecommerce", "mobile_app", "bank_transfer"]},
            "night(23-4)": D.transaction_hour.isin([23, 0, 1, 2, 3, 4])}
Str, Ste = segs(Dtr), segs(Dte); ctr, cte = Dtr.transaction_hour.isin(CLEAN).values, Dte.transaction_hour.isin(CLEAN).values
rows = []
for k in Str:
    st, se = np.asarray(Str[k], bool), np.asarray(Ste[k], bool)
    if k == "night(23-4)":   # night hours contain no clean hours -> use overall base level
        r = None
    c_s, o_s = (se & cte).sum(), (se & ~cte).sum()
    r_s = (st & ~ctr).sum() / max((st & ctr).sum(), 1)            # train other/clean ratio for this segment (hour dependence)
    inj = o_s - r_s * c_s; frac = inj / max(o_s + c_s, 1)
    rows.append((k, se.sum(), c_s, o_s, round(r_s, 2), round(inj), round(frac, 3), round(np.log(max(1 - frac, 1e-3)), 2), round(tr.fraud.values[st].mean(), 4)))
df = pd.DataFrame(rows, columns=["segment", "n_test", "clean_hr", "other_hr", "train_ratio", "est_injected", "inj_frac", "shift_if_inj_legit", "train_fraud_rate"])
pd.set_option("display.width", 200); print(df.to_string(index=False)); df.to_csv("experiments/inject_segments.csv", index=False)
