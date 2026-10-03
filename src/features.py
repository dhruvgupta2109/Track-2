import numpy as np, pandas as pd
CAT_MED_AMT = None  # filled by fit_constants

def fit_constants(tr):
    return {"cat_med_amt": tr.groupby("merchant_category").transaction_amount.median().to_dict(),
            "global_med_amt": float(tr.transaction_amount.median())}

def make_features(df, K):
    df = df.copy()
    amt, sp, t24, t1, age = df.transaction_amount, df.spend_last_24h, df.transactions_last_24h, df.transactions_last_1h, df.account_age
    med = df.merchant_category.map(K["cat_med_amt"]).astype(float).fillna(K["global_med_amt"])
    df["log_amt"] = np.log1p(amt)
    df["amt_rel_cat"] = np.log1p(amt) - np.log1p(med)
    df["log_spend"] = np.log1p(sp)
    df["avg_spend"] = np.log1p(sp / t24)
    df["amt_rel_avg"] = np.log1p(amt) - np.log1p(sp / t24)
    df["log_age"] = np.log1p(age)
    df["young"] = (age < 90).astype(float).where(age.notna())
    df["t1_hi"] = np.clip(t1 - 2, 0, None)
    df["t24_hi"] = np.clip(t24 - 9, 0, None)
    df["burst"] = (t24 >= 16).astype(float).where(t24.notna())
    df["t1_share"] = t1 / t24
    df["night"] = df.transaction_hour.isin([23, 0, 1, 2, 3]).astype(float)
    df["foreign"] = (df.country != "SG").astype(float).where(df.country.notna())
    df["n_missing"] = df[["merchant_category","country","transaction_channel","transactions_last_24h","spend_last_24h","account_age","new_device","transactions_last_1h"]].isna().sum(1)
    return df
