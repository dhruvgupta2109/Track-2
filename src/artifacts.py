"""Internal-consistency artefacts by population: train legit/fraud, test clean hours (pure base), test night band, test day band."""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
pd.set_option("display.width", 250)
tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv")
CATS = sorted(tr.merchant_category.dropna().unique())
def Xs(d):
    D = pd.get_dummies(d.merchant_category).reindex(columns=CATS, fill_value=0).values
    return np.c_[np.log1p(d.transactions_last_24h.fillna(5)), np.log(d.transaction_amount), D]
ok = tr.dropna(subset=["spend_last_24h", "transactions_last_24h", "merchant_category"])
L = ok[ok.fraud == 0]; m = LinearRegression().fit(Xs(L), np.log1p(L.spend_last_24h))
def stats(d):
    o = d.dropna(subset=["spend_last_24h", "transactions_last_24h", "merchant_category"])
    r = np.log1p(o.spend_last_24h) - m.predict(Xs(o))
    c = np.round(d.transaction_amount * 100) % 100; sc = np.round(d.spend_last_24h * 100) % 100
    t1 = d.transactions_last_1h; t24 = d.transactions_last_24h
    return {"n": len(d), "res_mean": r.mean(), "res_sd": r.std(), "res_q05": np.quantile(r, .05), "res_q95": np.quantile(r, .95),
            "c%10==0": (c % 10 == 0).mean(), "c==0": (c == 0).mean(), "sc==0": (sc == 0).mean(), "spend==0": (d.spend_last_24h == 0).mean(),
            "t24==1": (t24 == 1).mean(), "t1>t24": (t1 > t24).mean(), "t1==0": (t1 == 0).mean(), "amt<10": (d.transaction_amount < 10).mean(),
            "age%30==0": (d.account_age % 30 == 0).mean(), "age<30": (d.account_age < 30).mean(), "spend/amt<1": (d.spend_last_24h < d.transaction_amount).mean()}
h = te.transaction_hour
P = {"tr_legit": tr[tr.fraud == 0], "tr_fraud": tr[tr.fraud == 1], "tr_clean": tr[tr.transaction_hour.isin([5, 6, 7, 21, 22])],
     "te_clean": te[h.isin([5, 6, 7, 21, 22])], "te_night": te[h.isin([23, 0, 1, 2, 3, 4])], "te_day": te[h.between(8, 20)]}
S = pd.DataFrame({k: stats(v) for k, v in P.items()}).T
# implied injected-only stats: band = base share * clean + inj share * injected (base per hour = clean per hour)
base_ph = len(P["te_clean"]) / 5
for band, nh in [("te_night", 6), ("te_day", 13)]:
    nb = base_ph * nh; ni = len(P[band]) - nb
    S.loc[band.replace("te_", "inj_")] = (S.loc[band] * len(P[band]) - S.loc["te_clean"] * nb) / ni
    S.loc[band.replace("te_", "inj_"), "n"] = ni
print(S.round(4).to_string())
