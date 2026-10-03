"""Which segments crowd the top of the TEST ranking relative to where TRAIN frauds sit in the train ranking?
If P(x|fraud) is unchanged, excess share of a segment in the test top-k must be legitimate rows."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from features_prod import make_features; from constants import fit_constants
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv"); K = fit_constants(tr); Dtr, Dte = make_features(tr, K), make_features(te, K)
y = tr.fraud.values
oof = np.load("experiments/oof5_v5_prod.npy").mean(0); w2 = pd.read_csv("submissions/shift/W2_burst-1.0_ndold-0.75.csv").prediction.values
def segs(D):
    a = D.account_age; nd = D.new_device == 1; b = D.burst == 1
    return {"burst": b, "nd_old": nd & (a > 365) & ~b, "nd_young": nd & (a <= 365), "young<90": a < 90, "age90-365": (a >= 90) & (a <= 365),
            "foreign": D.foreign == 1, "night": D.night == 1, "amt>500": D.transaction_amount > 500, "amt_rel_cat>1": D.amt_rel_cat > 1,
            "t1h>=3": D.transactions_last_1h >= 3, "t24 10-15": (D.transactions_last_24h >= 10) & (D.transactions_last_24h < 16),
            "cash": D.merchant_category == "cash_transfer", "electronics": D.merchant_category == "electronics", "luxury": D.merchant_category == "luxury",
            "travel": D.merchant_category == "travel", "ecommerce": D.transaction_channel == "ecommerce", "bank_transfer": D.transaction_channel == "bank_transfer",
            "mobile_app": D.transaction_channel == "mobile_app", "card_present": D.transaction_channel == "card_present",
            "any_missing": D[["merchant_category", "country", "transaction_channel", "transactions_last_24h", "spend_last_24h", "account_age", "new_device", "transactions_last_1h"]].isna().any(axis=1),
            "spend>2000": D.spend_last_24h > 2000, "ID/PH/VN": D.country.isin(["ID", "PH", "VN"])}
Str, Ste = segs(Dtr), segs(Dte); rows = []
for k in Str:
    st, se = np.asarray(Str[k], bool), np.asarray(Ste[k], bool)
    r = {"segment": k, "train_all": st.mean(), "test_all": se.mean(), "train_fraud": st[y == 1].mean()}
    for q in [0.01, 0.03, 0.05]:
        ttr = oof >= np.quantile(oof, 1 - q); tte = w2 >= np.quantile(w2, 1 - q)
        r[f"trainTop{int(q*100)}"] = st[ttr].mean(); r[f"trainTop{int(q*100)}_fraud"] = st[ttr & (y == 1)].mean() if (ttr & (y == 1)).any() else np.nan
        r[f"testTop{int(q*100)}"] = se[tte].mean()
    rows.append(r)
df = pd.DataFrame(rows); df["excess_top3"] = df.testTop3 / df.trainTop3; df["excess_top5"] = df.testTop5 / df.trainTop5
pd.set_option("display.width", 250); print(df.round(3).sort_values("excess_top3", ascending=False).to_string(index=False))
